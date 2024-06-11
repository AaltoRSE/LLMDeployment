# Local Deployment of Open Source Large Language Models (LLMs)

This repo provides the basis of a system to deploy local LLM and provide a common gateway and access scheme to those models.

## Features

- Login scheme using SAML. Requires a SAML IdP (description to set up a local SAML provider for testing in the gateway repository)
- OpenAI compatible API based on llama-cpp-python specification
- API Key managment for all authorized users
- Inference using llama-cpp-python (i.e. allowing all features of llama-cpp-python)
- Horizontal scaling based on HTTP-Plugin for KEDA.
- Container recipies and Kubernetes config for easy deployment
- Helm chart for simple addition of new LLM Models

## Structure

The repo contains the following parts:

- `gateway`: The endpoint visible to the outside world, detailed description in the repo
- `inference`: Helm charts and Docker recipies for LLM inference pods.

## Requirements

To deploy the gateway and inference pods, you will need:

- A Kubernetes cluster with sufficient resources (GPUs are a must for good inference speed)
- A basic understanding of how to manage Kubernetes
