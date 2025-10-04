# Molmo Multi-Modal Demo

## How to run
```shell
docker compose up
```
The demo will run on CPU by default and will be available at [http://localhost:7860](http://localhost:7860). See [docker-compose.yml](./docker-compose.yml) for details on how to run on GPU.
Model weights will be stored in a docker volume mapping to the default location of huggingface cache: `~/.cache/huggingface/` on the local machine to prevent re-downloading. 

## Description
- Model: [MolmoE-1B-0924 from AllenAI](https://huggingface.co/allenai/MolmoE-1B-0924)
- Technical report: [ArXiv](https://arxiv.org/abs/2409.17146)

This demo showcases two capabilities of the Molmo model:

- **Pointing**: Identify and locate objects in images.
- **Image Description**: Generate detailed descriptions of images.

## Screenshots

![Pointing](./screenshots/pointing_cat.png)
![Description](./screenshots/description_meme.png)