# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# This file is used from below url. Additional thing added in this file to use IMDSv2 if IMDSv1 return no data
# https://github.com/aws/deep-learning-containers/blob/master/src/deep_learning_container.py

import sys
# move all pytorch related paths to the back of the path variable to avoid wrong import
# e.g. python3.9/site-packages/torch/types.py vs python3.9/types.py
i, j = 0, len(sys.path)-1
while i < j:
    if "torch" in sys.path[i]:
        sys.path[i], sys.path[j] = sys.path[j], sys.path[i]
        j -= 1
    i += 1

import argparse
import re
import json
import logging
import multiprocessing
import os
import signal

import botocore.session
import requests
import subprocess

TIMEOUT_SECS = 5


def _validate_instance_id(instance_id):
    """
    Validate instance ID
    """
    instance_id_regex = r"^(i-\S{17})"
    compiled_regex = re.compile(instance_id_regex)
    match = compiled_regex.match(instance_id)

    if not match:
        return None

    return match.group(1)


def _retrieve_instance_id():
    """
    Retrieve instance ID from instance metadata service
    """
    instance_id = None

    # Try IMDSv1
    instance_url = "http://169.254.169.254/latest/meta-data/instance-id"
    response = requests_helper(instance_url, timeout=0.1)

    if response is not None and not (400 <= response.status_code < 600):
        instance_id = _validate_instance_id(response.text)
    else:
        # Try IMDSv2
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
        token = None
        url = "http://169.254.169.254/latest/api/token"
        response = requests_put_helper(url, headers=headers)

        if response is not None and not (400 <= response.status_code < 600):
            token = response.text

        if token:
            headers={"X-aws-ec2-metadata-token": token}
            response = requests_helper(instance_url, headers=headers)

            if response is not None and not (400 <= response.status_code < 600):
                instance_id = _validate_instance_id(response.text)

    return instance_id


def _retrieve_instance_region():
    """
    Retrieve instance region from instance metadata service
    """
    region = None
    valid_regions = [
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-south-1",
        "ca-central-1",
        "eu-central-1",
        "eu-north-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "sa-east-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ]

    region_url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
    response = requests_helper(region_url, timeout=0.1)

    if response is not None and not (400 <= response.status_code < 600):
        response_json = json.loads(response.text)

        if response_json["region"] in valid_regions:
            region = response_json["region"]
    else:
        # Try IMDSv2
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
        token = None
        url = "http://169.254.169.254/latest/api/token"
        response = requests_put_helper(url, headers=headers)

        if response is not None and not (400 <= response.status_code < 600):
            token = response.text

        if token:
            headers={"X-aws-ec2-metadata-token": token}
            response = requests_helper(region_url, headers=headers)

            if response is not None and not (400 <= response.status_code < 600):
                response_json = json.loads(response.text)

                if response_json["region"] in valid_regions:
                    region = response_json["region"]

    return region


def _retrieve_device():
    device_type = "cpu"

    try:
        subprocess.check_output("nvidia-smi", shell=True, executable="/bin/bash")
        device_type = "gpu"
    except subprocess.CalledProcessError:
        pass

    if os.path.exists("/usr/local/bin/tensorflow_model_server_neuron"):
        device_type = "neuron"
    
    if os.path.isdir("/opt/ei_tools"):
        device_type = "eia"

    return device_type


def _retrieve_cuda():
    cuda_version = ""
    try:
        cuda_path = os.path.basename(os.readlink("/usr/local/cuda"))
        cuda_version_search = re.search(r"\d+\.\d+", cuda_path)
        cuda_version = "" if not cuda_version_search else cuda_version_search.group()
    except Exception as e:
        logging.error(f"Failed to get cuda path: {e}")
    return cuda_version


def _retrieve_os():
    version = ""
    name = ""
    with open("/etc/os-release", "r") as f:
        for line in f.readlines():
            if re.match(r"^ID=\w+$", line):
                name = re.search(r"^ID=(\w+)$", line).group(1)
            if re.match(r'^VERSION_ID="\d+\.\d+"$', line):
                version = re.search(r'^VERSION_ID="(\d+\.\d+)"$', line).group(1)
    return name + version


def requests_helper(url, timeout=0.1, headers=None):
    response = None
    try:
        if headers:
            response = requests.get(url, headers=headers, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        logging.error("Request exception: {}".format(e))

    return response

def requests_put_helper(url, timeout=0.1, headers=None):
    response = None
    try:
        if headers:
            response = requests.put(url, headers=headers, timeout=timeout)
        else:
            response = requests.put(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        logging.error("Request exception: {}".format(e))

    return response


def parse_args():
    """
    Parsing function to parse input arguments.
    Return: args, which containers parsed input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework", choices=["tensorflow", "mxnet", "pytorch"], help="framework of environment.", required=True
    )
    parser.add_argument("--framework-version", help="framework version of environment.", required=True)
    parser.add_argument(
        "--img-type",
        choices=["training", "inference", "dlami", "ami", "docker"],
        help="values of image. Two keywords training and inference belongs to dlc.",
        required=True,
    )
    parser.add_argument(
        "--pkg-type",
        choices=["conda", "pip"],
        help="package type like conda and pip",
        required=True,
    )

    args, _unknown = parser.parse_known_args()

    fw_version_pattern = r"\d+(\.\d+){1,2}(-rc\d)?"

    # PT 1.10 and above has +cpu or +cu113 string, so handle accordingly
    if args.framework == "pytorch":
        pt_fw_version_pattern = r"(\d+(\.\d+){1,2}(-rc\d)?)((\+cpu)|(\+cu\d{3})|(a0\+git\w{7}))"
        pt_fw_version_match = re.fullmatch(pt_fw_version_pattern, args.framework_version)
        if pt_fw_version_match:
            args.framework_version = pt_fw_version_match.group(1)
    assert re.fullmatch(fw_version_pattern, args.framework_version), (
        f"args.framework_version = {args.framework_version} does not match {fw_version_pattern}\n"
        f"Please specify framework version as X.Y.Z or X.Y."
    )

    return args


def query_bucket():
    """
    GET request on an empty object from an Amazon S3 bucket
    """
    response = None
    instance_id = _retrieve_instance_id()
    region = _retrieve_instance_region()
    args = parse_args()
    framework, framework_version, img_type, pkg_type = args.framework, args.framework_version, args.img_type, args.pkg_type
    py_version = sys.version.split(" ")[0]

    if instance_id is not None and region is not None:
        url = (
            "https://aws-deep-learning-containers-{0}.s3.{0}.amazonaws.com"
            "/dlc-containers-{1}.txt?x-instance-id={1}&x-framework={2}&x-framework_version={3}&x-py_version={4}&x-img_type={5}&x-pkg_type={6}".format(
                region, instance_id, framework, framework_version, py_version, img_type, pkg_type
            )
        )
        response = requests_helper(url, timeout=0.2)
        if os.environ.get("TEST_MODE") == str(1):
            with open(os.path.join(os.sep, "tmp", "test_request.txt"), "w+") as rf:
                rf.write(url)

    logging.debug("Query bucket finished: {}".format(response))

    return response


def tag_instance():
    """
    Apply instance tag on the instance that is running the container using botocore
    """
    instance_id = _retrieve_instance_id()
    region = _retrieve_instance_region()
    args = parse_args()
    framework, framework_version, img_type, pkg_type = args.framework, args.framework_version, args.img_type, args.pkg_type
    py_version = sys.version.split(" ")[0]
    device = _retrieve_device()
    cuda_version = f"_cuda{_retrieve_cuda()}" if device == "gpu" else ""
    os_version = _retrieve_os()

    tag = f"{framework}_{img_type}_{framework_version}_python{py_version}_{device}{cuda_version}_{os_version}_{pkg_type}"
    
    if "ami" in img_type:
        tag_struct = {"Key": "aws-dlami-autogenerated-tag-do-not-delete", "Value": tag}
    else:
        tag_struct = {"Key": "aws-dlc-autogenerated-tag-do-not-delete", "Value": tag}

    request_status = None
    if instance_id and region:
        try:
            session = botocore.session.get_session()
            ec2_client = session.create_client("ec2", region_name=region)
            response = ec2_client.create_tags(Resources=[instance_id], Tags=[tag_struct])
            request_status = response.get("ResponseMetadata").get("HTTPStatusCode")
            if os.environ.get("TEST_MODE") == str(1):
                with open(os.path.join(os.sep, "tmp", "test_tag_request.txt"), "w+") as rf:
                    rf.write(json.dumps(tag_struct, indent=4))
        except Exception as e:
            logging.error(f"Error. {e}")
        logging.debug("Instance tagged successfully: {}".format(request_status))
    else:
        logging.error("Failed to retrieve instance_id or region")

    return request_status


def main():
    """
    Invoke bucket query
    """
    # Logs are not necessary for normal run. Remove this line while debugging.
    logging.getLogger().disabled = True

    logging.basicConfig(level=logging.ERROR)

    bucket_process = multiprocessing.Process(target=query_bucket)
    tag_process = multiprocessing.Process(target=tag_instance)

    bucket_process.start()
    tag_process.start()

    tag_process.join(TIMEOUT_SECS)
    bucket_process.join(TIMEOUT_SECS)

    if tag_process.is_alive():
        os.kill(tag_process.pid, signal.SIGKILL)
        tag_process.join()
    if bucket_process.is_alive():
        os.kill(bucket_process.pid, signal.SIGKILL)
        bucket_process.join()


if __name__ == "__main__":
    main()
