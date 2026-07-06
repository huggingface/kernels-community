import unittest

import pr_autolabel as autolabel


def pr(title, body="", user="contributor", additions=1):
    return {
        "title": title,
        "body": body,
        "user": {"login": user},
        "additions": additions,
    }


def files(*paths):
    return [{"filename": path} for path in paths]


def backends(tax, labels):
    return labels & tax.by_dim["backend"]


class PrAutolabelHeuristicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = autolabel.load_taxonomy(".github/pr-labels.json")

    def labels_for(self, pr_data, *paths):
        return autolabel.finalize_labels(
            self.tax,
            autolabel.infer_labels(self.tax, pr_data, files(*paths)),
        )

    def test_dependabot_workflow_pr_ignores_body_semantics(self):
        labels = self.labels_for(
            pr(
                "build(deps): bump the actions group",
                body="Upstream sync notes from a generated dependency changelog.",
                user="dependabot[bot]",
            ),
            ".github/workflows/test_kernel_freshness_lookup.yml",
            ".github/workflows/build.yaml",
        )

        self.assertIn("deps", labels)
        self.assertNotIn("upstream-sync", labels)

    def test_rocm_ck_pr_is_not_triton_from_benchmark_body(self):
        labels = self.labels_for(
            pr(
                "Add aiter-flash-attn-ck: compiled CK FlashAttention (ROCm, with sinks)",
                body="Benchmark table compares CK vs Triton and SDPA.",
                additions=507401,
            ),
            "aiter-flash-attn-ck/build.toml",
            "aiter-flash-attn-ck/aiter/cpp_itfs/mha_fwd.hip",
            "aiter-flash-attn-ck/aiter/include/attention_common.cuh",
            "aiter-flash-attn-ck/ck/fmha/fmha_fwd.hpp",
        )

        self.assertIn("feature", labels)
        self.assertIn("rocm", labels)
        self.assertIn("new-kernel", labels)
        self.assertNotIn("triton", labels)

    def test_backend_specific_change_overrides_package_backend_defaults(self):
        labels = self.labels_for(
            pr("feat: prefer torch stable abi for cuda fa2"),
            "flash-attn2/build.toml",
            "flash-attn2/flash_attn/src/flash.h",
            "flash-attn2/flash_attn/src/flash_fwd_kernel.h",
            "flash-attn2/torch-ext/torch_binding_stable.cpp",
        )

        self.assertIn("cuda", labels)
        self.assertIn("abi-migration", labels)
        self.assertIn("build", labels)
        self.assertNotIn("cpu", labels)
        self.assertNotIn("xpu", labels)

    def test_repo_automation_title_intent_beats_template_words(self):
        labels = self.labels_for(
            pr(
                "scripts/check_kernel_freshness: include kernel_dir in stale report items",
                body="The current report mentions the same upstream URL. New test passes after the fix.",
            ),
            "scripts/check_kernel_freshness.py",
            "scripts/test_kernel_freshness_lookup.py",
        )

        self.assertIn("fix", labels)
        self.assertNotIn("upstream-sync", labels)

    def test_performs_does_not_imply_performance(self):
        labels = self.labels_for(
            pr(
                "feat: refactor kernel bot and add tests",
                body="execute_plan performs I/O and workflow dispatches.",
            ),
            ".github/scripts/dispatch.py",
            ".github/scripts/test_dispatch.py",
        )

        self.assertIn("feature", labels)
        self.assertNotIn("performance", labels)

    def test_vendoring_dependency_is_build_type_without_llm(self):
        labels = self.labels_for(
            pr("mamba-ssm: vendor causal_conv1d to remove dependency"),
            "mamba-ssm/build.toml",
            "mamba-ssm/causal-conv1d/causal_conv1d.cpp",
            "mamba-ssm/torch-ext/mamba_ssm/ops/triton/ssd_combined.py",
        )

        self.assertIn("build", labels)
        self.assertIn("vendoring", labels)

    def test_backend_labels_are_capped_to_top_priority(self):
        # finegrained-fp8 declares cuda/rocm/xpu in build.toml and its Python
        # files use Triton -- four backends. The cap keeps the two highest
        # priority (cuda, rocm) rather than flooding the PR.
        labels = self.labels_for(
            pr("finegrained-fp8: add swiglu support"),
            "finegrained-fp8/tests/test_moe.py",
            "finegrained-fp8/torch-ext/finegrained_fp8/batched.py",
            "finegrained-fp8/torch-ext/finegrained_fp8/fused_batched.py",
            "finegrained-fp8/torch-ext/finegrained_fp8/grouped.py",
        )

        self.assertEqual(backends(self.tax, labels), {"cuda", "rocm"})

    def test_global_cap_keeps_type_and_highest_priority_and_all_status(self):
        # A hand-built over-full set: 1 type + 2 backend + 2 semantic (5
        # descriptive) plus a status label. The global cap trims descriptive
        # labels to max_labels (keeping the highest priority ones) while the
        # uncapped status label always survives.
        labels = autolabel.finalize_labels(
            self.tax,
            {
                "feature",
                "cuda",
                "rocm",
                "new-kernel",
                "performance",
                "needs-rebase",
            },
        )

        descriptive = {l for l in labels if self.tax.dim_for(l) in self.tax.capped_dims}
        self.assertLessEqual(len(descriptive), self.tax.max_labels)
        self.assertIn("feature", labels)  # type always survives the cap
        self.assertIn("needs-rebase", labels)  # status is uncapped
        self.assertIn("new-kernel", labels)  # higher priority than performance
        self.assertNotIn("performance", labels)  # trimmed by the global cap


if __name__ == "__main__":
    unittest.main()
