FROM continuumio/miniconda3:latest
COPY ./RainyDay_Env.yml /RainyDay_Env.yml
COPY ./Source/RainyDay_utilities_Py3/RainyDay_functions.py RainyDay_utilities_Py3/RainyDay_functions.py
COPY ./Source/RainyDay_Py3.py /RainyDay_Py3.py
WORKDIR /
RUN conda env create -f /RainyDay_Env.yml -n RainyDay_Env
ENTRYPOINT ["conda","run","--no-capture-output","-n","RainyDay_Env","python3","/RainyDay_Py3.py"]
CMD []

# Using conda simplifies the build process significanlty but causes the image to take longer to build
# Conda has prebuilt binaries whereas pip would require us to install system libraries seprately which
# could lead to dependency conflicts.