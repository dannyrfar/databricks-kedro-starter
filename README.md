# The Databricks Kedro starter

This tutorial will walk through using this starter on a Databricks cluster on AWS, Azure or GCP.

>If you are using [Databricks Repos](https://docs.databricks.com/repos/index.html) to run a Kedro project then you should disable file-based logging in Kedro. This prevents Kedro from attempting to write to the read-only file system.

## Prerequisites

* New or existing [AWS](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/), [Azure](https://signup.azure.com) or [GCP](https://cloud.google.com) Databricks account with administrative privileges (can be on workspace or metastore level)
* Active [Databricks deployment](https://docs.databricks.com/getting-started/index.html) (Databricks Community Edition won't suffice as it doesn't allow you to provision personal tokens)
* Python(3.7+) and [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/) installed on your local machine
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed on your local machine
* An account on [GitHub](https://github.com/) (free tier or above), [Azure DevOps](https://azure.microsoft.com/en-us/products/devops) or [GitLab](https://about.gitlab.com)

>For a full list of supported Git providers check this [Databricks Repos](https://docs.databricks.com/repos/index.html#supported-git-providers) page

## Running Kedro project from a Databricks notebook

As noted in [this post describing CI/CD automation on Databricks](https://databricks.com/blog/2020/06/05/automate-continuous-integration-and-continuous-delivery-on-databricks-using-databricks-labs-ci-cd-templates.html#toc-2), _"Users may find themselves struggling to keep up with the numerous notebooks containing the ETL, data science experimentation, dashboards etc."_

Therefore, we do not recommend that you rely on the notebooks for running and/or deploying your Kedro pipelines unless it is unavoidable. The workflow described in this readme may be useful for experimentation and initial code development and analysis stages, but it is _not_ designed for productionisation.


### 1. Project setup

First, let's create a new virtual environment and, within it, a new Kedro project:

#### Using Conda
```bash
# create fresh virtual env
# NOTE: minor Python version of the environment
# must match the version on the Databricks cluster
conda create --name iris_databricks python=3.9 -y
conda activate iris_databricks

# install Kedro and create a new project
pip install "kedro~=0.18.7"
# name your project Iris Databricks when prompted for it
kedro new -s https://github.com/dannyrfar/databricks-kedro-starter --checkout main
```

#### Using virtualenv
```bash
# create fresh virtual env
# NOTE: minor Python version of the environment
# must match the version on the Databricks cluster.
# Make sure to install python and point to
# the install location in the PYTHON_PATH variable
export PYTHON_PATH=/usr/local/bin/python3.9
pip install virtualenv
# create a new virtual environment in .venv inside the project folder
virtualenv iris-databricks/.venv -p $PYTHON_PATH
source iris-databricks/.venv/bin/activate

# install Kedro and create a new project
pip install "kedro~=0.18.7"
# name your project Iris Databricks when prompted for it
kedro new -s https://github.com/dannyrfar/databricks-kedro-starter --checkout main
```

### 2. Install dependencies and run locally

Now, as the project has been successfully created, we should move into the project root directory, install project dependencies, and then start a local test run using [Spark local execution mode](https://stackoverflow.com/a/54064507/3364156), which means that all Spark jobs will be executed in a single JVM locally, rather than in a cluster. The `Databricks Kedro starter` used to generate the project already has all necessary configuration for it to work, you just need to have `pyspark` Python package installed, which is done for you by `pip install -r src/requirements.txt` or `poetry install` command below.

If you are using `poetry`:

```bash
# install poetry if you will manage dependencies in poetry
pip install poetry
# change the directory to the project root
cd iris-databricks/
# compile and install the project dependencies, this may take a few minutes
poetry install
# start a local run
python -m kedro run
```

If you are using `setup.py/requiements.txt`:

```bash
# change the directory to the project root
cd iris-databricks/
# compile and install the project dependencies, this may take a few minutes
pip install -r src/requirements.txt
# start a local run
python -m kedro run
```

You should get a similar output:
```console
...
2023-03-21 19:42:37,916 - kedro.io.data_catalog - INFO - Saving data to 'iris_features' (ManagedTableDataSet)...
2023-03-21 19:42:44,435 - kedro.runner.sequential_runner - INFO - Completed 2 out of 3 tasks
2023-03-21 19:42:44,435 - kedro.io.data_catalog - INFO - Loading data from 'iris_features' (ManagedTableDataSet)...
2023-03-21 19:42:44,573 - kedro.pipeline.node - INFO - Running node: generate_predictions([iris_features]) -> [iris_predictions]
0.94642
2023-03-21 19:42:51,029 - kedro.io.data_catalog - INFO - Saving data to 'iris_predictions' (ManagedTableDataSet)...
2023-03-21 19:42:56,304 - kedro.runner.sequential_runner - INFO - Completed 3 out of 3 tasks
2023-03-21 19:42:56,304 - kedro.runner.sequential_runner - INFO - Pipeline execution completed successfully.
```
### 3. Create a Databricks cluster

If you already have an active cluster with runtime version `11.3 LTS` or above, you can skip this step and move onto [Step 4](#4-create-repository-personal-access-token-and-add-to-databricks). Here is [how to find clusters in your Databricks workspace](https://docs.databricks.com/clusters/clusters-manage.html).

Follow the [Databricks official guide to create a new cluster](https://docs.databricks.com/clusters/create-cluster.html). For the purpose of this tutorial (and to minimise costs) we recommend the following settings:
* Runtime: `11.3 LTS (Scala 2.12, Spark 3.3.0)`
* Access mode: `Single user` or `No isolation shared`
* Worker type: `Standard_DS3_v2` (Azure) | `m4.large` (AWS) | `n1-standard-4` (GCP)
* Driver Type: `Same as worker`
* Workers: `1`
* Enable spot instances
* Terminate after: `30 minutes`
* Disable autoscaling

>While your cluster is being provisioned, you can continue to the next step to setup your repo in Databricks.

As a result you should have:
* A Kedro project, which runs with the local version of PySpark library
* A running Databricks interactive cluster

### 4. Create and add repository personal access token to Databricks

To synchronise the project between the local development environment and Databricks, we will use a private GitHub/Azure DevOps/GitLab repository, which you will create in the next step. For authentication, we will need a GitHub/Azure DevOps/GitLab personal access token, so go ahead and create this token in your [GitHub](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token), [Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Windows), or [GitLab](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) developer settings. If you are using another Git provider, follow the instructions for your platform to generate a personal access token.

>Make sure that `repo` scopes are enabled for your token.

Once you have the token, you will add it to your Databricks workspace. 
1. Under `User Settings` 
2. Navigate to `Git integration`
3. Select your Git provider
4. Fill in your Git username 
5. Paste the the [personal access token](#4-create-github-personal-access-token) you setup earlier.

![](assets/images/databricks_token.png)

>If you do not see this section, reach out to your Databricks administrator to enable Repos in your workspace.

### 5. Create a (private) repository for your Kedro project and push your code

Now you should create a new repository in [GitHub](https://docs.github.com/en/github/getting-started-with-github/create-a-repo), [Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/repos/git/create-new-repo), [GitLab](https://docs.gitlab.com/ee/user/project/index.html#create-a-project) using the official guides. You can keep the repository private and you don't need to commit to it just yet.

To connect to the newly created repository locally, you can use one of 2 options:

* **SSH:** If you choose to connect with SSH, you will also need to configure the SSH connection to [GitHub](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh), [Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/repos/git/use-ssh-keys-to-authenticate), or [GitLab](https://docs.gitlab.com/ee/user/ssh.html). You might already have an [existing SSH key configured for GitHub](https://docs.github.com/en/github/authenticating-to-github/checking-for-existing-ssh-keys) or your Git provider.
* **HTTPS:** If using HTTPS, you will be asked for your username and password when you push your first commit - please use your git username and your [personal access token](#4-create-github-personal-access-token) generated in the previous step as a password.

We will use a CLI to push the newly created Kedro project to your newly created repository. First, you need to initialise Git in your project root directory:

```bash
# change the directory to the project root
cd iris-databricks/
# initialise git
git init
```

Then, create the first commit:

```bash
# add all files to git staging area
git add .
# create the first commit
git commit -m "first commit"
```

Finally, push the commit to your repository. Follow these instructions for getting the SSH/HTTPS URLs for your repository: [GitHub](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories), [Azure DevOps](https://learn.microsoft.com/en-us/azure/devops/repos/git/clone), or [GitLab](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#clone-a-repository). Fill in the `<HTTPS URL>` or `<SSH URL>` according to your method of authenticating with git:

```bash
# configure a new remote
# for HTTPS run:
git remote add origin <HTTPS URL>
# or for SSH run:
git remote add origin <SSH URL>

# verify the new remote URL
git remote -v

# push the first commit
git push --set-upstream origin main
```

### 6. Configure Databricks to Read from your Repo

The project has now been pushed to your private repository, and in order to pull it from the Databricks, we need to add the repo to your workspace.

[Log into your Databricks workspace](https://docs.databricks.com/workspace/workspace-details.html#workspace-instance-names-urls-and-ids) and then:
1. Open `Repos` tab
2. Click on `Add Repo`
3. Paste the URL of your repo (HTTPS)
4. Click on `Create Repo`

![](assets/images/databricks_repo.png)

You should now be able to browse your repository inside Databricks under your email.

### 7. Run your Kedro project from the Databricks notebook

Congratulations, you are now ready to run your Kedro project from Databricks!

You will find a notebook in the notebooks folder of the databricks starter called `sample_run.ipynb.py` that contains all these cells and can be ran as is on Databricks. To create a new notebook follow the steps below. The `project_root` is the location of the repo in your workspace in the format: `/Workspace/Repos/{user email}/{repo name}`.

You can interact with Kedro in Databricks through the Kedro [IPython extension](https://ipython.readthedocs.io/en/stable/config/extensions/index.html), `kedro.ipython`.

The Kedro IPython extension launches a [Kedro session](https://docs.kedro.org/en/stable/kedro_project_setup/session.html) and makes available the useful Kedro variables `catalog`, `context`, `pipelines` and `session`. It also provides the `%reload_kedro` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) that reloads these variables (for example, if you need to update `catalog` following changes to your Data Catalog).

The IPython extension can be used in a Databricks notebook in a similar way to how it is used in [Jupyter notebooks](https://docs.kedro.org/en/stable/notebooks_and_ipython/kedro_and_notebooks.html).

[Create your Databricks notebook](https://docs.databricks.com/notebooks/notebooks-manage.html#create-a-notebook) and remember to attach it to the cluster you have just configured.

In your newly-created notebook, put each of the below code snippets into a separate cell, then [run all cells](https://docs.databricks.com/notebooks/run-notebook.html):

* Install Kedro and the latest compatible version of Kedro-Datasets.

```console
%pip install "kedro==0.18.7" "kedro-datasets[databricks.ManagedTableDataSet]~=1.3.0" kedro-viz
```

* Instantiate the Kedro globals using the magic

```python
%load_ext kedro.ipython
```

```python
%reload_kedro <project_root>
```

* Run Kedro project

```python
session.run()
```

You should get a similar output:

```console
...
2023-03-21 19:42:37,916 - kedro.io.data_catalog - INFO - Saving data to 'iris_features' (ManagedTableDataSet)...
2023-03-21 19:42:44,435 - kedro.runner.sequential_runner - INFO - Completed 2 out of 3 tasks
2023-03-21 19:42:44,435 - kedro.io.data_catalog - INFO - Loading data from 'iris_features' (ManagedTableDataSet)...
2023-03-21 19:42:44,573 - kedro.pipeline.node - INFO - Running node: generate_predictions([iris_features]) -> [iris_predictions]
0.94642
2023-03-21 19:42:51,029 - kedro.io.data_catalog - INFO - Saving data to 'iris_predictions' (ManagedTableDataSet)...
2023-03-21 19:42:56,304 - kedro.runner.sequential_runner - INFO - Completed 3 out of 3 tasks
2023-03-21 19:42:56,304 - kedro.runner.sequential_runner - INFO - Pipeline execution completed successfully.
```

### 8. Running Kedro-Viz on Databricks

For Kedro-Viz to run with your Kedro project, you need to ensure that both the packages are installed in the same scope (notebook-scoped vs. cluster library). i.e. if you `%pip install kedro` from inside your notebook then you should also `%pip install kedro-viz` from inside your notebook.
If your cluster comes with Kedro installed on it as a library already then you should also add Kedro-Viz as a [cluster library](https://docs.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries).

Kedro-Viz can then be launched in a new browser tab with the `%run_viz` line magic:
```ipython
%run_viz
```

### 9. How to use datasets stored on Databricks DBFS

DBFS is a distributed file system mounted into a DataBricks workspace and accessible on a DataBricks cluster. It maps cloud object storage URIs to relative paths so as to simplify the process of persisting files. With DBFS, libraries can read from or write to distributed storage as if it's a local file.
To use datasets with DBFS, the file path passed to the dataset **must** be prefixed with `/dbfs/` and look something like, `/dbfs/example_project/data/02_intermediate/processed_data`. This applies to all datasets, including `SparkDataSet`.
> **Note**: Most Python code, except PySpark, will try to resolve a file path in the driver node storage by default, this will result in an `DataSetError` if the code is using a file path that is actually a DBFS save location. To avoid this, always make sure to point the file path to `/dbfs` when storing or loading data on DBFS. For more rules on what is saved in DBFS versus driver node storage by default, please refer to the [Databricks documentation](https://docs.databricks.com/files/index.html#what-is-the-root-path-for-databricks).

This does not apply to datasets that do not use files.


