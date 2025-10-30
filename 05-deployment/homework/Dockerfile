FROM agrigorev/zoomcamp-model:2025

# Install uv library in Docker 
RUN pip install --no-cache-dir uv

# Copy the uv files into our working directory 
COPY ["pyproject.toml", "uv.lock", "./"]

# install the uv dependencies for the project and deploy them.
RUN uv sync --frozen

# Copy any python files and the model we had to the working directory of Docker 
COPY ["predict_web_service_v2.py", "./"]

# We need to expose the 8000 port because we're not able to communicate with Docker outside it
EXPOSE 8000

# If we run the Docker image, we want our prediction app to be running
ENTRYPOINT ["uv", "run", "uvicorn", "predict_web_service_v2:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]