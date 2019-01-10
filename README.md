# <span style="color:#e3c59e">bu</span>
User-Centric Semi-Structured Natural Language Generation. I'm not naming this "improved-waddle" Github...

## Huh, looks like I neglected this README for a while
Anyway, I'm just here to explain the structure for this repo, since it's a lil' bit different to what I'd normally do. So, you see that little folder called [aibu](aibu/)? Yeah, that's where all the [source code](aibu/src) for this boi is kept. Well that and <span style="color:#e3c59e">**bu**</span>'s [Dockerfile](aibu/Dockerfile). Odd right? Well. Yeah. It is. BUT, I have a _"good"_ reason for it. Kinda. I'm essentially future-proofing this, because that's always a **_fantastic_** idea. I'm not sure whether I'll add [distbu](https://github.com/VictorIJnr/distbu) to this repo - well as much as you can have it as a sub-repo that is - because it would be easier to work with and debug with a cheeky lil' `docker-compose up -d --build` you know? But yeah, that's why we have this structure.  
Granted, I could decide that, even though they're both important to <span style="color:#e3c59e">**bu**</span>, I'll have them as separate repos. <span style="color:#333">distbu</span> could spin off to become their own thing for people to upload and pull datasets (because that would actually be incredibly useful) but for now, I don't know whether I'll leave them as separate repos (slightly like this option more) or put them into one big repo. 

You know what? Forget it. Forget what you just read. Sure, I could take out those paragraphs, but where's the fun in that? This repo is for **bu** and nothing else. distbu can do their own thing in their own repo. They're fine. They're a big boi. They can handle it.

## Running bu
Okay, so this is a lil' bit more complex than some "hurr durr clone the repo and run my bash script I'm a genius blah di bla bla" nonsense. No. This is more demanding than some crappy C++ game your nephew Timmy made. Kid can't code whatsoever.  

Follow these steps and everything will be absolutely dandy. Don't worry, you can trust me. Everything here's safe, I'm like, 85% sure, that's pretty much 100% so I'd fancy your odds here ngl. 
1. Alright, well just like Timmy's trash game, you need to clone the repo.
   + Honestly, you should've seen that coming. I don't want to hear any excuses as to why you haven't done that yet.
2. Make sure you're okay. Clarify that the `git clone` worked for you. If you downloaded this as a .zip file then God help you. You probably use Emacs over Vim too. Distgusting. 
3. Build the Dockerfile by running `docker image build -t bu .`
   + Oh yeah I forgot to mention you'll want to have docker on your machine to run this. Well, it's not a pre-requisite, you could use this just fine without it, but it's nice and I know that everything's fine :D.
4. That takes a while to build doesn't it?  
5. Ignore steps 3 and 4.
   + Yep, step 4 was pretty much a buffer between 3 and 5 :P
6. Don't get mad at me. Weren't you taught to read all of the instructions before following them?
7. Pull my pre-built docker image by running `docker image pull victorijnr/bu:latest`
   + By the time you see this, it should be made public so you can also go [here](https://cloud.docker.com/repository/docker/victorijnr/bu/tags) to see all the different image tags. Just stick to the latest imo, older builds may not be compatible with the current state of this repo.
8. Run the command `docker container run -it --name bu --mount src=distbu,dst=/bu/ --mount src=$(pwd)/aibu/src,type=bind,dst=/bu/src victorijnr/bu:neg-1.0` from this repo's root folder.
    + Just copy-paste that, I shouldn't really need to tell you that, it's a pretty long one
    + Don't worry, I'm learning bash in the meantime so this'll just be something like "Run ./imabashscript"
9. Oh wait I realised you don't have the distbu volume. That sucks. Okay, if distbu isn't up and running by the time you see this, I'll have extra instructions on setting up the distbu volume for the last step in a later section.
10. Play around!
    + You'll have a bash shell in front of you; if you've never used docker before, you're now inside the container. It's like if a VM went on a diet and could share data with other VMs. Nothing scary.
    + Use the shell to poke around the source code or any data I'm using. If I've finished distbu, you won't have a data/ folder to play around with.
    + But you can run something basic like `python -m characterisation.stack` from the src/ directory to see what it's like to filter the original dataset. Head's up though, it's not exactly exciting.
