# <span style="color:#e3c59e">bu</span>
I [pivoted](https://www.youtube.com/watch?v=M1vfXoUNDYA). This was initially meant to be dynamic, user-centric natural language generation. I know, I know, that's kinda ambitious in itself. But I legitimately thought I could still do it up until the beginning of 2019. So what, about halfway through the year? Yeah, about right; I had a little under 2 months until the initial deadline at that point. 

But that didn't work out like I hoped. That's why <span style="color:#e3c59e">**bu**</span> was converted to "cluster" users with one another based on their stylometry. It's not **_really_** clustering, but it's close enough. I call it _User Stylometry Association_. It's a much better name than the "improved-waddle" recommended by Github...

## Running bu
Honestly, I didn't make this public so you could run it yourself. My paper's [out there](./SH_Report.pdf) so you can read that instead. But this is meant to highlight the code I wrote. Even then it's pretty sloppy.  

So, this is a bit of a mess to run. Sorry. I spent too long tweaking stuff like hyper-parameters for the convolutional autoencoder or running experiments for classification.  
There's some stuff that this is still missing though. The most important one is the lack of [distbu](https://github.com/VictorIJnr/distbu/). I didn't complete it in time for my submission so this code doesn't use it. But distbu's done now, so I **_may_** come back to this and integrate it with distbu. But across the next couple of steps and the extra ones in the report (but you shouldn't need those) you can run <span style="color:#e3c59e">**bu**</span> just fine. 
<sup style="font-size:0.1em">I'm lazy and have a couple more READMEs to fine-tune so I'm just leaving the old instructions I had here.</sup>

So yeah, follow these steps and everything will be absolutely dandy. Don't worry, you can trust me. Everything here's safe, I'm like, 85% sure, that's pretty much 100% so I'd fancy your odds here ngl. 
1. Alright, well just like any trash Instagram clone on GitHub, you need to clone the repo.
   + Honestly, you should've seen that coming. I don't want to hear any excuses as to why you haven't done that yet.
2. Make sure you're okay. Clarify that the `git clone` worked for you. If you downloaded this as a .zip file then God help you. You probably use Emacs over Vim too. Distgusting. 
3. Build the Dockerfile by running `docker image build -t bu .`
   + Oh yeah I forgot to mention you'll want to have docker on your machine to run this. Well, it's not a pre-requisite, you could use this just fine without it, but it's nice and it just makes everything okay. You know? :D
4. That takes a while to build doesn't it?  
5. Ignore steps 3 and 4.
   + Yep, step 4 was pretty much a buffer between 3 and 5 :P
6. Don't get mad at me. Weren't you taught to read all of the instructions before following them?
7. Pull my pre-built docker image by running `docker image pull victorijnr/bu:latest`
   + By the time you see this, it should be made public so you can also go [here](https://cloud.docker.com/repository/docker/victorijnr/bu/tags) to see all the different image tags. Just stick to the latest one imo, older builds may not be compatible with the current state of this repo.
8. Run the command `docker container run -it --name bu --mount src=distbu,dst=/bu/ --mount src=$(pwd)/src,type=bind,dst=/bu/src victorijnr/bu:neg-1.0` from this repo's root folder.
    + Just copy-paste that, I shouldn't really need to tell you that, it's a pretty long one
    + Don't worry, I'm learning bash in the meantime so this'll just be something like "Run `./imabashscript`"
9. Oh wait I realised you don't have the distbu volume. That sucks. Okay, if distbu isn't up and running by the time you see this, I'll have extra instructions on setting up the distbu volume for the last step in a later section.
10. Play around!
    + You'll have a bash shell in front of you; if you've never used docker before, you're now inside the container. It's like if a VM went on a diet and could share data with other VMs. Nothing scary.
    + Use the shell to poke around the source code or any data I'm using. If I've finished distbu, you won't have a data/ folder to play around with.
    + But you can run something basic like `python -m characterisation.stack` from the src/ directory to see what it's like to filter the original dataset. Head's up though, it's not exactly exciting.

## Demoing bu
Okay, so this section is pretty much just for me to show people that bu works. Feel free to read this though, especially if you're using this in the most barebones way possible. Like, we're talking no pre-trained models and just a bunch of data barebones here. In that case, your a damn mental case. Plus how're you doing that anyway? This doesn't have distbu nor a data folder? <span style="font-size: 0.01em">The fuck?</span> Nevermind, you do you. If you want, download some data from [here](https://archive.org/download/stackexchange). But you'll want it as a CSV. I converterd it all programmatically in [here](./src/characterisation/helpers/XMLParse.py) so that'd help you out.

### Classification
I'm going to start from the beginning. You think that'll be obvious, but I could very easily ignore the initial step(s) and be just fine.  
I won't go low-level into the actual commands needed for demonstration, that was for the last section. This is going to be pretty high-level until the last couple of bits.

Alright, you'll want to get yourself a dataset to work with. **_BUT_** it has to be one of the [StackExchange Datasets](https://archive.org/details/stackexchange). Yep. Just go for the _worldbuilding_ dataset. Plus download the _serverfault_ dataset so the Convolutional AutoEncoder can train itself. <sup style="font-size:0.4em">I can't be bothered to parameterise its training yet.</sup>  

We good? Good. Now I'm going to assume you're using [docker](https://www.docker.com/), if not, [get it](https://www.docker.com/get-started). Why? Because I don't want to set up a virtual environment.  
Take your downloaded data and **_yeet_** it into a docker volume; make sure you name it `distbu` as that's the name I use, and that's the name used in my scripts. You'll need to run those scripts in order to set up the container.

Alright, so now you have the container set up, we'll be in a position to demonstrate the capabilities of bu.

### Transformation
So, I haven't gotten to the part where bu performs "Natural Language Transformation" (I guess?) so I don't even know where I would start when it comes to demoing that. So for now, this'll remain blank because I don't know what to say. 

Oh, but I just realised how much of a minefield Natural Language Transformation could become. Looks like this'll be fun. But, later. 