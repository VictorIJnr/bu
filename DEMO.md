## Running the project
1. Pull my pre-built docker image by running `docker image pull victorijnr/bu:latest`
   + By the time you see this, it should be made public so you can also go [here](https://cloud.docker.com/repository/docker/victorijnr/bu/tags) to see all the different image tags. Ideally stick to the latest one, older builds may not be compatible with the current state of this project.
   + This can be run without the use of docker as long as the requirements are installed (as seen in the requirements.txt file) along with a version of Python 3.6. In that case ignore every other step apart from Step 3.
2. Alternatively, build the Dockerfile by running `docker image build -t bu .`
3. Create a docker volume called **_distbu_** (or anything else as long as you change the src in the next step appropriately)
   + Place the data file inside this docker volume. On Linux, the path should be akin to `/var/lib/docker/volumes/<volume name>`
   + The volume's (and any sub-folders) permissions should be such that the logged in user can read and write to all the data files (and folders). 
   + In the event that docker is not being used, eveything should be fine if the **_data_** folder is kept in the project's root directory.

4. Run the command `docker container run -it --name bu --mount src=distbu,dst=/bu/ --mount src=$(pwd)/src,type=bind,dst=/bu/src victorijnr/bu:neg-1.0` from this repo's root folder.
5.  Play around!
    + If docker is being used, you'll have a bash shell in front of you; you're now inside the container. 
    + Use the shell to poke around the source code or any data I'm using. 
    + Some commands to run:
      + `python -m characterisation.stack`
        + Filters the default (worldbuilding) dataset to only contain Posts (and comments) where users appear at least 5 times.
      + `python -m characterisation.classibu --train --folds 3`
        + To train a support vector machine model with 3 fold cross-validation
        + This one command hides a vast amount of the work being done. Including data filtering, dimension reduction, hyper-parameter optimisation etc.


