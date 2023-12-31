How to Run the Code

Hey there! Here's a quick guide to get you started with the codebase:
Step 1: Set Up Your Environment

After cloning the directory and opening it in VSCode, you'll need to set up a virtual environment. This helps in managing dependencies and keeps things tidy. Run the following command:

bash

python3 -m venv venv

Step 2: Activate the Virtual Environment

Next, activate the virtual environment. This step is crucial to ensure that all the dependencies are correctly installed in this isolated environment. Run:

bash

source ./venv/bin/activate

Step 3: Install Dependencies

Now, let's install all the required dependencies listed in the requirements.txt file. Run:

bash

pip install -r requirements.txt

Step 4: Run the Main Script

Finally, to see the magic happen, run the main script:

bash

python3 ./main.py

If all goes well, you'll see a large JSON output on your screen. That's a sign that everything is set up correctly!
How to Write an Environment

To create your own environment, you'll need to dive into some documentation. Don't worry, it's not as daunting as it sounds! Check out the Ray RLlib Environments Documentation. It's simpler than other interfaces and should get you up to speed quickly.
How to Learn RLlib and Ray

Depending on how deep you want to go, there are three levels of commitment you can choose from:
Minimum Effort

For a quick start, read through the entire RLlib documentation. It's concise and should take about an hour. Focus on the Environment section since that's what we'll be working with mostly. Check it out here: RLlib Documentation.
Medium Effort

If you have more time, I recommend going through the entire Ray documentation. This will take about 2-4 hours but will give you a comprehensive understanding of the library. It's worth it if you're planning to dive deeper into our project. Here's the link: Ray Overview.
Maximum Effort

And for those who really want to master this library, there's a whole book on Ray! It's quite an undertaking, but by the end of it, you'll be an expert. The book is beginner-friendly and builds up from the basics. However, it's a significant time investment. If you're up for it, here's the book: Learning Ray (PDF).

Also, feel free to peek at the env/tiger.py file where I've implemented the tiger environment. Just a heads up, it's a bit rough around the edges and not fully documented yet.
Project Plan

Here's what we're aiming to achieve:

 Create and test environments like box pushing and rock sampling, specifically using MultiAgentEnv.
 Develop a function to convert a MultiAgentEnv to a Gym Env.
 Learn how to effectively collect metrics from a run.

     Conduct the experiments and gather all the necessary metrics.

Let's get started!