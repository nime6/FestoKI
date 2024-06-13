# FestoKI - Quality Assurance in Manufacturing Processes


## About
A brief description of what your project does and why it’s useful.

## Getting Started

```bash
git clone https://github.com/nime6/FestoKI.git
cd FestoKI
```

### Docker Setup

Install Docker Engine (https://docs.docker.com/engine/install/)

Build Docker image from the Dockerfile.

```bash
docker build -t festo_image .

# -t: Is used to tag the Docker image with a name and optionally a tag in the format name:tag
```

Run image from Dockerfile in an interactive shell

```bash
docker run -it --rm --name festo_container festo_image /bin/bash

# -it: Runs container in interactive mode, with /bin/bash in an interactive shell
# --rm: Ensures that the container is removed as soon as it stops running.
```



