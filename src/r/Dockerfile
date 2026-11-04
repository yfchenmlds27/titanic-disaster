FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install R packages first (better layer caching)
COPY src/r/install_packages.R /app/src/r/install_packages.R
RUN Rscript /app/src/r/install_packages.R

# Copy the code (the data is mounted at runtime, not baked into the image)
COPY src/r/ /app/src/r/

CMD ["Rscript", "src/r/titanic.R"]
