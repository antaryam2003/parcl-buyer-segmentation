# Deploying the dashboard to Streamlit Community Cloud

The repository is deployment-ready: `requirements.txt`, `.streamlit/config.toml` and the pre-computed model outputs in `outputs/` are all committed, so the app boots without running the pipeline.

Takes about three minutes.

## Steps

1. Go to **<https://share.streamlit.io>** and click **Sign in with GitHub**. Use the same GitHub account that owns the repository.

2. Authorise Streamlit when GitHub asks. It needs read access to the repository.

3. Click **Create app**, then choose **Deploy a public app from GitHub**.

4. Fill in the three fields:

   | Field | Value |
   |---|---|
   | Repository | `antaryam2003/parcl-buyer-segmentation` |
   | Branch | `main` |
   | Main file path | `dashboard/app.py` |

5. Optionally click **Advanced settings** and set **Python version** to `3.12`. The app works on 3.11–3.13; pinning avoids a surprise if Streamlit changes its default.

6. Click **Deploy**. The first build installs the dependencies and takes two to four minutes. The app URL will look like:

   ```
   https://parcl-buyer-segmentation.streamlit.app
   ```

7. Once the app loads, confirm the header KPIs read **2,000 buyers · $2.52B capital committed · 7,305 units**. If they do, the deployment is correct.

That URL is what goes in the **Deployed project link** field of the submission form.

## If the build fails

**`ModuleNotFoundError`** — confirm `requirements.txt` is at the repository root and lists the missing package. Then use **Manage app → Reboot**.

**"outputs/segmented_clients.csv is missing"** — the `outputs/` directory was not committed. Check that `.gitignore` does not exclude it, then commit and push:

```bash
git add -f outputs/
git commit -m "Add model outputs required by the dashboard"
git push
```

**Build times out** — make sure you are deploying with `requirements.txt` and not `requirements-dev.txt`; the dev file pulls in Jupyter, which is unnecessary for the dashboard and slow to install.

**App sleeps after inactivity** — normal on the free tier. It wakes on the next visit in about 30 seconds. Open the link once shortly before submitting so it is warm for a reviewer.

## Updating after a change

Streamlit Cloud watches the branch and redeploys on every push:

```bash
git add -A
git commit -m "Describe the change"
git push
```

If a change to the analysis alters the numbers, re-run the pipeline first so the committed outputs match the paper:

```bash
python run_pipeline.py
python tools/build_paper.py
git add -A && git commit -m "Refresh model outputs and paper" && git push
```
