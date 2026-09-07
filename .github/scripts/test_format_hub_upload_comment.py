import json

import pytest

from format_hub_upload_comment import upload_lines


@pytest.mark.parametrize("nested", [False, True])
def test_upload_lines_discovers_json_in_directory(tmp_path, nested):
    upload_dir = tmp_path / "hub-uploads"
    if nested:
        upload_dir /= "hub-upload-linux"
    upload_dir.mkdir(parents=True)
    (upload_dir / "upload.json").write_text(
        json.dumps(
            {
                "pull_requests": [
                    {"url": "https://huggingface.co/kernels/example/discussions/1"}
                ]
            }
        )
    )

    assert upload_lines("example", "kernels-staging", [str(tmp_path / "hub-uploads")]) == [
        "- Hub repo: https://huggingface.co/kernels/kernels-staging/example",
        "- Hub pull request: https://huggingface.co/kernels/example/discussions/1",
    ]


def test_upload_lines_still_accepts_inline_json():
    upload = json.dumps(
        {
            "pull_requests": [
                {"url": "https://huggingface.co/kernels/example/discussions/1"}
            ]
        }
    )

    assert upload_lines("example", "kernels-community", [upload]) == [
        "- Hub repo: https://huggingface.co/kernels/kernels-community/example",
        "- Hub pull request: https://huggingface.co/kernels/example/discussions/1",
    ]
