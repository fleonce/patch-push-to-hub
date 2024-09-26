import os
import warnings
from pathlib import Path
from typing import Optional, Union, List

from with_argparse import with_argparse
from transformers.utils import PushToHubMixin, working_or_temp_dir


def _resolve_import(class_name):
    if "." not in class_name:
        raise ValueError(f"Cannot import {class_name}")

    parent_package = class_name.split(".")[:-1]
    parent_package = ".".join(parent_package)
    class_name = class_name.split(".")[-1]

    module = __import__(parent_package, fromlist=[class_name])
    return getattr(module, class_name)

@with_argparse
def upload(
    model_type: str,
    model_name: str,
    upload_name: str,
    public: bool = False,
    readme: Optional[Path] = None,
):
    model_class = _resolve_import(model_type)
    model = model_class.from_pretrained(model_name)
    model.save_pretrained("models/" + upload_name)

    if readme and readme.exists():
        with (
            readme.open("r") as f_in,
            open("models/" + upload_name + "/README.md") as f_out,
        ):
            f_out.write(f_in.read())

    model = model_class.from_pretrained("models/" + upload_name)
    patch_push_to_hub(
        model,
        repo_id=upload_name,
        private=not public,
        use_temp_dir=True,
    )


def patch_push_to_hub(
    self: PushToHubMixin,
    repo_id: str,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = None,
    private: Optional[bool] = None,
    token: Optional[Union[bool, str]] = None,
    max_shard_size: Optional[Union[int, str]] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = None,
    tags: Optional[List[str]] = None,
    **deprecated_kwargs,
) -> str:
    """
    Upload the {object_files} to the 🤗 Model Hub.

    Parameters:
        self (`PushToHubMixin`):
            The model/tokenizer/etc in question
        repo_id (`str`):
            The name of the repository you want to push your {object} to. It should contain your organization name
            when pushing to a given organization.
        use_temp_dir (`bool`, *optional*):
            Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
            Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
        commit_message (`str`, *optional*):
            Message to commit while pushing. Will default to `"Upload {object}"`.
        private (`bool`, *optional*):
            Whether or not the repository created should be private.
        token (`bool` or `str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
            is not specified.
        max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
            Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
            will then be each of size lower than this size. If expressed as a string, needs to be digits followed
            by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
            Google Colab instances without any CPU OOM issues.
        create_pr (`bool`, *optional*, defaults to `False`):
            Whether or not to create a PR with the uploaded files or directly commit.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether or not to convert the model weights in safetensors format for safer serialization.
        revision (`str`, *optional*):
            Branch to push the uploaded files to.
        commit_description (`str`, *optional*):
            The description of the commit that will be created
        tags (`List[str]`, *optional*):
            List of tags to push on the Hub.

    Examples:

    ```python
    from transformers import {object_class}

    {object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

    # Push the {object} to your namespace with the name "my-finetuned-bert".
    {object}.push_to_hub("my-finetuned-bert")

    # Push the {object} to an organization with the name "my-finetuned-bert".
    {object}.push_to_hub("huggingface/my-finetuned-bert")
    ```
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
    if repo_path_or_name is not None:
        # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
        # repo_id from the folder path, if it exists.
        warnings.warn(
            "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
            "`repo_id` instead.",
            FutureWarning,
        )
        if repo_id is not None:
            raise ValueError(
                "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
            )
        if os.path.isdir(repo_path_or_name):
            # repo_path: infer repo_id from the path
            repo_id = repo_id.split(os.path.sep)[-1]
            working_dir = repo_id
        else:
            # repo_name: use it as repo_id
            repo_id = repo_path_or_name
            working_dir = repo_id.split("/")[-1]
    else:
        # Repo_id is passed correctly: infer working_dir from it
        working_dir = repo_id.split("/")[-1]

    # Deprecation warning will be sent after for repo_url and organization
    repo_url = deprecated_kwargs.pop("repo_url", None)
    organization = deprecated_kwargs.pop("organization", None)

    repo_id = self._create_repo(
        repo_id, private=private, token=token, repo_url=repo_url, organization=organization
    )

    if use_temp_dir is None:
        use_temp_dir = not os.path.isdir(working_dir)

    with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
        files_timestamps = self._get_files_timestamps(work_dir)

        readme_path = os.path.join(work_dir, "README.md")
        if not os.path.exists(readme_path):
            raise ValueError("No README.md found in '" + work_dir + "'")

        # Save all files.
        self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

        return self._upload_modified_files(
            work_dir,
            repo_id,
            files_timestamps,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            revision=revision,
            commit_description=commit_description,
        )
