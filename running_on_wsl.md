# Run the stack on WSL

Note that for the stack to work on Windows, you should have running version on ollama installed somehow. Since Windows, is not yet supported, we can only use WSL.

I won't cover the activation of WSL procedure which you will find very easy on internet. Assuming that you have a version of WSL on your local windows machine and a running docker Desktop software installer and running as well, you can follow the following steps:

1. enable docker-desktop to use WSL using the following tutorial:

>To connect WSL to Docker Desktop, you need to follow the instructions below:
>
>1. First, ensure that you have installed Docker Desktop for Windows on your machine. If you haven't, you can download it from ¹.
>2. Next, open Docker Desktop and navigate to **Settings**.
>3. From the **Settings** menu, select **Resources** and then click on **WSL Integration**.
>4. On the **WSL Integration** page, you will see a list of available WSL distributions. Select the distribution that you want to connect to Docker Desktop.
>5. Once you have selected the distribution, click on **Apply & Restart**.
>
>After following these steps, your WSL distribution should be connected to Docker Desktop.
>
>
>1.  Docker Desktop WSL 2 backend on Windows | Docker Docs. https://docs.docker.com/desktop/wsl/.
>2. Get started with Docker containers on WSL | Microsoft Learn. https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers.
>3. How to configure Docker Desktop to work with the WSL. https://tutorials.releaseworksacademy.com/learn/how-to-configure-docker-desktop-to-work-with-the-wsl.html.
>4. How can I access wsl2 which is used by Docker desktop?. https://stackoverflow.com/questions/70449927/how-can-i-access-wsl2-which-is-used-by-docker-desktop.

After the activation enter into the WSL with the command `wsl` and type `docker`.

2. Install ollama on WSL using https://github.com/jmorganca/ollama (avec la commande `curl https://ollama.ai/install.sh | sh`)
    - The script have been downloaded in `./install_ollama.sh` and you do the smae thing with `sh ./install_ollama.sh`
    - To list the downloaded model: `ollama list`. This command could lead to:
    ```sh
    NAME            ID              SIZE    MODIFIED       
    llama2:latest   7da22eda89ac    3.8 GB  22 minutes ago
    ```
    - (OPTIONAL) To remove model: `ollama rm llama2`
    - To run the ollama on WSL: `ollama run llama2`

3. clone the repo
4. cd into the repo and enter wsl
5. run `docker-compose up`

# Run the stack on MAC:

On MAC you can follow this tutorial: https://collabnix.com/getting-started-with-genai-stack-powered-with-docker-langchain-neo4j-and-ollama/