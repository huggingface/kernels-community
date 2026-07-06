import unittest

import pr_autolabel as autolabel


def pr(title, user="contributor"):
    return {"title": title, "user": {"login": user}}


class PrefilterTest(unittest.TestCase):
    def test_dependabot_author_is_deps(self):
        self.assertEqual(
            autolabel.prefilter_type(pr("bump x", user="dependabot[bot]")), "deps"
        )

    def test_build_deps_title_is_deps(self):
        self.assertEqual(
            autolabel.prefilter_type(pr("build(deps): bump the actions group")), "deps"
        )

    def test_conventional_prefix_maps_to_type(self):
        self.assertEqual(
            autolabel.prefilter_type(pr("feat: add rocm backend")), "feature"
        )
        self.assertEqual(
            autolabel.prefilter_type(pr("fix(core): correct stride")), "fix"
        )
        self.assertEqual(autolabel.prefilter_type(pr("docs: update CARD")), "docs")

    def test_non_conventional_title_defers_to_model(self):
        # kernel PRs often use "<kernel>: <desc>" which is not a conventional type.
        self.assertIsNone(
            autolabel.prefilter_type(pr("finegrained-fp8: add swiglu support"))
        )
        self.assertIsNone(
            autolabel.prefilter_type(pr("Support sycl-tla 0.9.1 barrier API"))
        )


class ClassifyLabelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = autolabel.load_taxonomy(".github/pr-labels.json")

    def test_validates_and_drops_unknown_labels(self):
        result = {
            "type": "fix",
            "backend": ["cuda", "bogus"],
            "semantic": ["new-kernel"],
        }
        self.assertEqual(
            autolabel.classify_labels(self.tax, result), {"fix", "cuda", "new-kernel"}
        )

    def test_single_select_string_is_accepted(self):
        self.assertEqual(
            autolabel.classify_labels(self.tax, {"type": "feature"}), {"feature"}
        )

    def test_none_or_junk_result_is_empty(self):
        self.assertEqual(autolabel.classify_labels(self.tax, None), set())
        self.assertEqual(
            autolabel.classify_labels(self.tax, {"backend": "not-a-list-item"}), set()
        )


class FinalizeLabelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = autolabel.load_taxonomy(".github/pr-labels.json")

    def test_type_fallback_when_missing(self):
        self.assertIn("chore", autolabel.finalize_labels(self.tax, {"cuda"}))

    def test_single_select_type_keeps_top_priority(self):
        out = autolabel.finalize_labels(self.tax, {"feature", "fix"})
        self.assertEqual(out & self.tax.by_dim["type"], {"feature"})  # authored first

    def test_backend_capped_to_two_highest_priority(self):
        out = autolabel.finalize_labels(
            self.tax, {"feature", "cuda", "rocm", "metal", "cpu"}
        )
        self.assertEqual(out & self.tax.by_dim["backend"], {"cuda", "rocm"})

    def test_global_cap_trims_descriptive_but_keeps_type_and_status(self):
        out = autolabel.finalize_labels(
            self.tax,
            {"feature", "cuda", "rocm", "new-kernel", "performance", "needs-rebase"},
        )
        descriptive = {l for l in out if self.tax.dim_for(l) in self.tax.capped_dims}
        self.assertLessEqual(len(descriptive), self.tax.max_labels)
        self.assertIn("feature", out)  # type always survives
        self.assertIn("needs-rebase", out)  # status is uncapped
        self.assertIn("new-kernel", out)  # higher priority than performance
        self.assertNotIn("performance", out)  # trimmed by the global cap


if __name__ == "__main__":
    unittest.main()
