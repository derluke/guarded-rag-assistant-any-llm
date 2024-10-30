# Guarded RAG Assistant

The guarded RAG assistant is an easily customizable recipe for building a RAG-powered chatbot. 

In addition to creating a hosted, shareable user interface, the guarded RAG assistant provides:

* Regex and prompt-injection guardrails.
* A predictive sidecar model that evaluates response quality.
* GenAI-focused custom metrics.
* DataRobot MLOps hosting, monitoring, and governing the individual backend deployments.

![Using the Guarded RAG Assistant](https://s3.amazonaws.com/datarobot_public/drx/recipe_gifs/launch_gifs/guardedraghq-small.gif)

## Setup


1. Clone the template repository.
   
   ```
   git clone https://github.com/datarobot-community/guarded-rag-assistant.git
   ```

2. Create the file `.env` in the root directory of the repo and populate your credentials.

   ```
   DATAROBOT_API_TOKEN=...
   DATAROBOT_ENDPOINT=...  # e.g. https://app.datarobot.com/api/v2
   OPENAI_API_KEY=...
   OPENAI_API_VERSION=...  # e.g. 2024-02-01
   OPENAI_API_BASE=...  # e.g. https://your_org.openai.azure.com/
   OPENAI_API_DEPLOYMENT_ID=...  # e.g. gpt-4
   PULUMI_CONFIG_PASSPHRASE=...  # required, choose an alphanumeric passphrase to be used for encrypting pulumi config
   ```
   
3. Set environment variables using your `.env` file. Use the helper script provided below:
   ```
   source set_env.sh
   # on Windows: set_env.bat or Set-Env.ps1
   ```
   This script exports environment variables from `.env` and activate the virtual 
   environment in `.venv/` (if present).

5. If you're a first-time user, install the Pulumi CLI by following the instructions [here](#details) before proceeding with this workflow.


6. Create a new stack for your project (update the placeholder `YOUR_PROJECT_NAME`).
   ```
   pulumi stack init YOUR_PROJECT_NAME
   ```

7. Provision all resources and install dependencies in a new virtual environment located in `.venv/`.
   ```
   pulumi up
   ```

### Details

Instructions for installing Pulumi are [here](https://www.pulumi.com/docs/iac/download-install/). In many cases this can be done
with the code below:

```
curl -fsSL https://get.pulumi.com | sh
```

Restart your terminal.

```
pulumi login --local

source set_env.sh
# on Windows: set_env.bat or Set-Env.ps1
```

Python must be installed for this project to run. By default, pulumi will use the Python binary aliased to `python3` to create a new virtual environment. If you wish to self-manage your virtual environment, delete the `virtualenv` and `toolchain` keys from `Pulumi.yaml` before running `pulumi up`. For projects that will be maintained, DataRobot recommends forking the repo so upstream fixes and improvements can be merged in the future.

### Feature flags

This app template requires certain feature flags to be enabled or disabled in your DataRobot account. The required feature flags can be found in [infra/feature_flag_requirements.yaml](infra/feature_flag_requirements.yaml). Contact your DataRobot representative or administrator for information on enabling the feature.

## Architecture Overview
![Guarded RAG Architecture](https://s3.amazonaws.com/datarobot_public/drx/recipe_gifs/rag_architecture.svg)

## Make changes

### Change the RAG documents

1. Replace `assets/datarobot_english_documentation_docsassist.zip` with a new zip file containing .pdf, .docx,
   .md, or .txt documents ([example alternative docs here](https://s3.amazonaws.com/datarobot_public_datasets/ai_accelerators/acme_corp_company_policies_source_business_victoria_templates.zip)).
3. Update the `rag_documents` setting in `infra/settings_main.py` to specify the local path to the
   new zip file.
4. Run `pulumi up` to update your stack.

### Change the LLM

1. Modify your `.env`.
   ```
   GOOGLE_SERVICE_ACCOUNT=''  # insert json service key between the single quotes, newlines are OK
   GOOGLE_REGION=...  # default is 'us-west1'
   ```
2. Update your environment and install `google-auth`.
   ```
   source set_env.sh
   pip install google-auth
   ```
3. Update the credential type to be provisioned in `infra/settings_llm_credential.py`.
   ```
   # credential = AzureOpenAICredentials()
   # credential.test()
   from docsassist.credentials import GoogleLLMCredentials
   credential = GoogleLLMCredentials()
   credential.test('gemini-1.5-flash-001')  # select a model for validating the credential
   ```
4. Configure a Gemini blueprint to be provisioned in `infra/settings_rag.py`.
   ```
   # llm_id=GlobalLLM.AZURE_OPENAI_GPT_3_5_TURBO,
   llm_id=GlobalLLM.GOOGLE_GEMINI_1_5_FLASH,
   ```
5. Run `pulumi up` to update your stack.
   
### Fully custom front-end
1. Edit `infra/settings_main.py` and update `application_type` to `ApplicationType.DIY`
   - Optionally, update `APP_LOCALE` in `docsassist/i18n.py` to toggle the language. 
     Supported locales include French (fr_FR), Spanish (es_LA), Korean (ko_KR), and 
     Brazilian Portuguese (pt_BR) in addition to the English default (en_US).
2. Run `pulumi up` to update your stack with the example custom Streamlit frontend,
3. After provisioning the stack at least once, you can also edit and test the Streamlit
   front-end locally using `streamlit run app.py` from the `frontend/` directory (don't 
   forget to initialize your environment using `source set_env.sh`).

### Fully custom RAG chunking, vectorization and retrieval
1. Install additional requirements (e.g. FAISS, HuggingFace).
   ```
   source set_env.sh
   pip install -r requirements-extra.txt
   ```
2. Edit `infra/settings_main.py` and update `rag_type` to `RAGType.DIY`.
3. Run `pulumi up` to update your stack with the example custom RAG logic.
4. Edit `data_science/build_rag.ipynb` to customize the doc chunking, vectorization logic.
5. Edit `deployment_diy_rag/custom.py` to customize the retrieval logic & LLM call.
6. Run `pulumi up` to update your stack.

## Share results

1. Log into the DataRobot application.
2. Navigate to **Registry > Applications**.
3. Navigate to the application you want to share, open the actions menu, and select **Share** from the dropdown.

## Delete all provisioned resources
```
pulumi down
```
