# Deploying the live demo to Hugging Face Spaces

The demo runs as a Docker Space: the FastAPI model service and the Streamlit dashboard
share one container. The trained model is stored in a separate Hugging Face model repo and
downloaded when the Space starts, so it never goes into git.

## One-time setup

1. **Create a Hugging Face account** at huggingface.co.
2. **Create an access token** under Settings → Access Tokens, with *write* permission.
3. **Add it to this GitHub repo** under Settings → Secrets and variables → Actions:
   - Secret `HF_TOKEN`: the token
   - Variable `HF_SPACE`: the Space to deploy to, e.g. `your-hf-username/visual-inspection`
   - Variable `HF_MODEL_REPO`: where the model lives, e.g. `your-hf-username/visual-inspection-model`

## After training

4. **Create the model repo** named in `HF_MODEL_REPO` (New → Model) and upload:

   ```
   patchcore/meta.json          from models/patchcore/<category>/
   patchcore/memory_bank.npy    from models/patchcore/<category>/
   samples/                     a few test images: some good, some defective
   README.md                    credit MVTec AD and its CC BY-NC-SA 4.0 licence
   ```

   Large files are stored with Git LFS automatically when uploaded through the website.

5. **Deploy:** in GitHub, open Actions → *Deploy demo to Hugging Face* → *Run workflow*.
   After that, every push that changes the app redeploys it.

The first start takes a few minutes: the image builds, then the backbone weights download
on the first inspection.
