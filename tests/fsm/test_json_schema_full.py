import json
import re

import pytest
import requests
import requests_cache
from referencing.exceptions import Unresolvable

from outlines.fsm.json_schema import build_regex_from_schema

requests_cache.install_cache("test_request_cache", expire_after=3600)


def get_json_schema_tests_from_repo(
    repo="json-schema-org/JSON-Schema-Test-Suite", configs_dir="tests/draft2020-12"
):
    api_url = f"https://api.github.com/repos/{repo}/contents/{configs_dir}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    contents = response.json()

    results = []
    for item in contents:
        if item["type"] == "file" and item["name"].endswith(".json"):
            file_url = item["download_url"]
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            json_data = file_response.json()

            for entry in json_data:
                for test in entry["tests"]:
                    results.append(
                        {
                            "file": item["name"],
                            "schema": json.dumps(entry["schema"]),
                            "data": json.dumps(test["data"]),
                            "is_valid": test["valid"],
                        }
                    )

    return results


# @pytest.mark.skip("Utility for improving compliance with json schema spec")
@pytest.mark.parametrize("sample", get_json_schema_tests_from_repo())
def test_json_schema_validity(sample):
    """
    Assert that we either correctly handle a schema, or raise NotImplementedError
    """
    try:
        pattern = build_regex_from_schema(sample["schema"])
    except (NotImplementedError, Unresolvable):
        return

    if sample["is_valid"]:
        assert re.fullmatch(pattern, sample["data"]), "Failed to match valid schema"
    else:
        assert (
            re.fullmatch(pattern, sample["data"]) is None
        ), "Incorrectly matched invalid schema"
