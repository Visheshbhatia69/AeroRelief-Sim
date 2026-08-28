# Publishing AeroRelief-Sim

This project can be published as a public Streamlit dashboard and linked from a poster QR code.

## Recommended Option: Streamlit Community Cloud

Use this option for the dissertation poster because it gives a clean public URL such as:

```text
https://aerorelief-sim.streamlit.app
```

## Files Needed In The GitHub Repository

Upload these files/folders:

```text
app.py
simulation.py
optimization.py
metrics.py
visualization.py
experiment.py
requirements.txt
README.md
data/
```

Do not upload:

```text
.venv/
__pycache__/
poster files
Word documents
PowerPoint files
old generated result images
log files
```

## Deployment Steps

1. Create a new GitHub repository, for example `aerorelief-sim`.
2. Upload the required project files listed above.
3. Make the GitHub repository public.
4. Go to `https://share.streamlit.io`.
5. Click **Create app** or **Deploy an app**.
6. Select the GitHub repository.
7. Set the main file path to:

```text
app.py
```

8. Choose a public app URL such as:

```text
aerorelief-sim
```

9. Deploy the app.
10. Open the public URL and test the dashboard.

## QR Code For Poster

After deployment, copy the public Streamlit URL and generate a QR code from it.

Recommended QR label for the poster:

```text
Try AeroRelief-Sim
Scan to open the interactive UAV placement simulator.
```

## If The App Is Slow Online

For public demo use, reduce the default simulation settings:

- users: 60 to 80;
- UAVs: 4 to 6;
- PSO iterations: 60 to 100;
- swarm size: 25 to 40.

This keeps the app responsive for people scanning the poster QR code.
