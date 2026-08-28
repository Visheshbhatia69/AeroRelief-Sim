# Deploying AeroRelief-Sim

**AeroRelief-Sim: A Priority-Aware UAV Placement Simulator for Disaster Communication**
is a Streamlit application. It can run locally for development or be deployed
to Streamlit Community Cloud for sharing through a public link.

## Streamlit Entry Point

Set the main application file to:

```text
app.py
```

## Example Public URL

The intended public URL format is:

```text
aerorelief-sim
```

which becomes:

```text
https://aerorelief-sim.streamlit.app
```

## Local Run

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

On Windows, the included launch script can also be used:

```text
run_dashboard.bat
```

## Cloud Deployment Steps

1. Open Streamlit Community Cloud.
2. Connect the GitHub repository.
3. Select the `main` branch.
4. Set the main file path to `app.py`.
5. Deploy the application.
6. Open the public URL and run a small test scenario.

## Poster QR Code

After deployment, the public Streamlit URL can be converted into a QR code.
A short label such as the following works well:

```text
Try AeroRelief-Sim
Scan to open the interactive UAV placement simulator.
```

## Runtime Notes

For public demonstrations, moderate simulation settings keep the dashboard
responsive:

- users: 60 to 80;
- UAVs: 4 to 6;
- PSO iterations: 60 to 100;
- swarm size: 25 to 40.
