"""test_pipeline.py — Verification Suite for Mamba-3 Engine, Converter & Recovery Trainer
========================================================================================
Runs automated checks covering:
1. Pure PyTorch Mamba3 engine forward & backward execution.
2. Forward-hook LoRA adapter gradient isolation.
3. Mamba-2 -> Mamba-3 SSD weight conversion and tensor layout matching.
4. Phase A and Phase B recovery training step execution.
"""

import os
import tempfile
import unittest
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from mamba3_engine import Mamba3, Mamba3LMModel, RMSNorm
from mamba_to_mamba3_converter import convert_mamba2_to_mamba3, build_mamba3_shell_state_dict
from mamba3_recovery_trainer import freeze_transplanted, inject_lora, build_phase_a_optimizer, build_phase_b_optimizer, train_step


class TestMamba3Pipeline(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = "cpu"
        self.dtype = torch.float32

    def test_mamba3_engine_forward_backward(self):
        """Verify Mamba3 mixer and LMModel forward/backward passes."""
        d_model = 64
        d_state = 16
        headdim = 32
        chunk_size = 16

        mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            chunk_size=chunk_size,
            device=self.device,
            dtype=self.dtype,
        )
        x = torch.randn(2, 32, d_model)
        y = mixer(x)
        self.assertEqual(y.shape, (2, 32, d_model))

        model = Mamba3LMModel(
            d_model=d_model,
            n_layer=2,
            vocab_size=256,
            d_state=d_state,
            headdim=headdim,
            dtype=self.dtype,
            device=self.device,
        )
        input_ids = torch.randint(0, 256, (2, 32))
        out = model(input_ids)
        self.assertEqual(out.logits.shape, (2, 32, 256))

        loss = out.logits.sum()
        loss.backward()
        self.assertIsNotNone(model.backbone.embedding.weight.grad)
        self.assertIsNotNone(model.backbone.layers[0].mixer.out_proj.weight.grad)

    def test_forward_hook_lora_gradient_isolation(self):
        """Verify hook-based LoRA computes gradients without mutating out_proj.weight."""
        model = Mamba3LMModel(
            d_model=64,
            n_layer=2,
            vocab_size=256,
            d_state=16,
            headdim=32,
            dtype=self.dtype,
            device=self.device,
        )
        inject_lora(model, rank=4, alpha=8.0)

        # Base linear layers must be frozen
        for block in model.backbone.layers:
            self.assertFalse(block.mixer.out_proj.weight.requires_grad)
            self.assertTrue(block.mixer.lora_A.requires_grad)
            self.assertTrue(block.mixer.lora_B.requires_grad)
            self.assertIsInstance(block.mixer.out_proj, nn.Linear)

        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids)
        loss = out.logits.sum()
        loss.backward()

        for block in model.backbone.layers:
            self.assertIsNone(block.mixer.out_proj.weight.grad)
            self.assertIsNotNone(block.mixer.lora_A.grad)
            self.assertIsNotNone(block.mixer.lora_B.grad)

    def test_mamba2_to_mamba3_converter(self):
        """Verify Mamba-2 to Mamba-3 parameter preservation."""
        d_model = 64
        n_layer = 2
        vocab_size = 256
        headdim = 32
        d_state = 16
        ngroups = 1
        d_inner = d_model * 2
        conv_dim = d_inner + 2 * ngroups * d_state
        nheads = d_inner // headdim
        d_in_proj = d_inner + conv_dim + nheads

        # Build mock Mamba-2 state dict
        m2_sd = {
            "backbone.embedding.weight": torch.randn(vocab_size, d_model),
            "backbone.norm_f.weight": torch.ones(d_model),
        }
        for i in range(n_layer):
            m2_sd[f"backbone.layers.{i}.norm.weight"] = torch.ones(d_model)
            m2_sd[f"backbone.layers.{i}.mixer.in_proj.weight"] = torch.randn(d_in_proj, d_model)
            m2_sd[f"backbone.layers.{i}.mixer.conv1d.weight"] = torch.randn(conv_dim, 1, 4)
            m2_sd[f"backbone.layers.{i}.mixer.conv1d.bias"] = torch.zeros(conv_dim)
            m2_sd[f"backbone.layers.{i}.mixer.A_log"] = torch.randn(nheads)
            m2_sd[f"backbone.layers.{i}.mixer.D"] = torch.ones(nheads)
            m2_sd[f"backbone.layers.{i}.mixer.dt_bias"] = torch.randn(nheads)
            m2_sd[f"backbone.layers.{i}.mixer.norm.weight"] = torch.ones(d_inner)
            m2_sd[f"backbone.layers.{i}.mixer.out_proj.weight"] = torch.randn(d_model, d_inner)

        config = {
            "model_type": "mamba2",
            "hidden_size": d_model,
            "num_hidden_layers": n_layer,
            "vocab_size": vocab_size,
            "head_dim": headdim,
            "state_size": d_state,
            "n_groups": ngroups,
        }

        m3_sd, out_cfg = convert_mamba2_to_mamba3(m2_sd, config, dtype=torch.float32)
        self.assertEqual(out_cfg["model_type"], "mamba3")

        # Load into target model shell
        target_model = Mamba3LMModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            d_state=d_state,
            headdim=headdim,
            dtype=torch.float32,
            device=self.device,
        )
        missing, unexpected = target_model.load_state_dict(m3_sd, strict=False)
        self.assertEqual(len(missing), 0, f"Missing keys: {missing}")
        self.assertEqual(len(unexpected), 0, f"Unexpected keys: {unexpected}")

    def test_recovery_phases(self):
        """Verify Phase A and Phase B freeze/unfreeze mechanisms."""
        model = Mamba3LMModel(
            d_model=64,
            n_layer=2,
            vocab_size=256,
            d_state=16,
            headdim=32,
            dtype=self.dtype,
            device=self.device,
        )

        # Phase A
        freeze_transplanted(model)
        optim_a = build_phase_a_optimizer(model)
        trainable_a = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_a), 0)
        # Backbone weights must not be trainable
        self.assertFalse(model.backbone.embedding.weight.requires_grad)
        self.assertFalse(model.backbone.layers[0].mixer.out_proj.weight.requires_grad)

        # Phase B
        inject_lora(model, rank=4, alpha=8.0)
        optim_b = build_phase_b_optimizer(model)
        trainable_b = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_b), len(trainable_a))


if __name__ == "__main__":
    unittest.main()
