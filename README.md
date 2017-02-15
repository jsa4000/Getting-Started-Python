#Visual Studio Code

This folder is a project done in Visual Studio Code and Python Language 3.5.1 |Anaconda 4.0.0 (64-bit)
The idea behind using Visual Studio Code is to simplify the importation of this project for any platforms or any IDE.

##1. Installation

In order to install Visual Studio Code you can download from the following link: https://code.visualstudio.com/docs/?dv=win

##2. Phyton

> For further information, go through the offical Web Page
> 	https://code.visualstudio.com/docs/languages/python

###2.1 Virtual Environment

First at all it's recommended to use **Virtual Environments** for Python, since you can use different versions or modules.

>virtualenv is a tool to create isolated Python environments. virtualenv creates a folder which contains all the necessary executables to use the packages that a Python project would need.

1. Install virtualenv via pip:

	pip install virtualenv

2- **Create** a virtual environment for a project:

	cd my_project_folder
	virtualenv venv

 virtualenv venv will create a folder in the current directory which will contain the Python executable files, and a copy of the pip library which you can use to install other packages. The name of the virtual environment (in this case, it was venv) can be anything; omitting the name will place the files in the current directory instead.

 This creates a copy of Python in whichever directory you ran the command in, placing it in a folder named venv.

 You can also use the **Python interpreter** of your choice (like python2.7).

	virtualenv -p /usr/bin/python2.7 venv

3- To begin using the virtual environment, it needs to be **activated**:

	source venv/bin/activate


###2.2 VS Code Extensions

There are several extension for Visual Studio code that simplify the coding:

- Magic Python
- Python
- Python Extended
- Python for VSCode

> You can type the name of the extension you are looking by typing ext install into the Command Palette Ctrl+Shift+P..
> Tip: Snippets appear in the same way as code completion Ctrl+Space.

###2.3 Configuration

> If this is the first time you open the "Task: Configure Task Runner", by pressing Ctrl+Shift+P.
> Finally, you need to select "other" at the bottom of the next selection list.


####2.3.1 Tasks

Task are running by pressing the Shorcut _Ctrl+Shift+B (where B denotes *"Build"*)

This will bring up the properties which you can then change to suit your preference. In this case you want to change the following properties;

- Change the Command property from "tsc" (TypeScript) to "Python"
- Change showOutput from "silent" to "Always"
- Change args (Arguments) from ["Helloworld.ts"] to ["${file}"] (filename)
- Delete the last property problemMatcher
- Save the changes made

This file will be stored in _.vscode\tasks.json_

	{
		// See https://go.microsoft.com/fwlink/?LinkId=733558
		// for the documentation about the tasks.json format
		"version": "0.1.0",
		"command": "${config.python.pythonPath}",
		"isShellCommand": true,
		"args": ["${file}"],
		"showOutput": "always"
	}

####2.3.2 Settings

> https://code.visualstudio.com/Docs/customization/userandworkspace

It's easy to configure VS Code the way you want by editing the various setting files where you will find a great number of settings to play with.

VS Code provides two different scopes for settings:

- **User** these settings apply globally to any instance of VS Code you open
- **Workspace** these settings are stored inside your workspace in a .vscode folder and only apply when the workspace is opened. Settings defined on this scope override the user scope.

Creating User and Workspace Settings

The menu under File > Preferences (Code > Preferences on Mac) provides entry to configure user and workspace settings. You are provided with a list of Default Settings. Copy any setting that you want to change to the related settings.json file.

In the example below, we disabled line numbers in the editor and configured line wrapping to wrap automatically based on the size of the editor.

	// Place your settings in this file to overwrite default and user settings.
	//"python.pythonPath": "python"
	//"python.pythonPath": "C:/Users/user/AppData/Local/Continuum/Anaconda2/Python.exe"
	{
			// Controls if lines should wrap. The lines will wrap at min(editor.wrappingColumn, viewportWidthInColumns).
			"editor.wordWrap": false,
			"python.pythonPath": "python"
	}

####2.3.3 Launch

> https://code.visualstudio.com/Docs/editor/debugging

To debug a simple app in VS Code, you simply have to press F5 and VS Code will try to debug your currently active file.

However, for advanced debugging you first need to set up your launch configuration file - launch.json. Click on the Configure gear icon on the Debug view top bar and VS Code will generate a launch.json file under your workspace's .vscode folder. VS Code will try to automatically detect your debug environment, if unsuccessful you will have to choose your debug environment manually.

	{
		"version": "0.2.0",
		"configurations": [
			{
				"name": "Python",
				"type": "python",
				"request": "launch",
				"stopOnEntry": true,
				"pythonPath": "${config.python.pythonPath}",
				"program": "${file}",
				"cwd": "${workspaceRoot}",
				"debugOptions": [
					"WaitOnAbnormalExit",
					"WaitOnNormalExit",
					"RedirectOutput"
				]
			},
			{
				"name": "Integrated Terminal/Console",
				"type": "python",
				"request": "launch",
				"stopOnEntry": true,
				"pythonPath": "${config.python.pythonPath}",
				"program": "${file}",
				"cwd": "null",
				"console": "integratedTerminal",
				"debugOptions": [
					"WaitOnAbnormalExit",
					"WaitOnNormalExit"
				]
			},
			{
				"name": "External Terminal/Console",
				"type": "python",
				"request": "launch",
				"stopOnEntry": true,
				"pythonPath": "${config.python.pythonPath}",
				"program": "${file}",
				"cwd": "null",
				"console": "externalTerminal",
				"debugOptions": [
					"WaitOnAbnormalExit",
					"WaitOnNormalExit"
				]
			},
			{
				"name": "Attach (Remote Debug)",
				"type": "python",
				"request": "attach",
				"localRoot": "${workspaceRoot}",
				"remoteRoot": "${workspaceRoot}",
				"port": 3000,
				"secret": "my_secret",
				"host": "localhost"
			}
		]
	}


##3. Git

> For further details check this link: https://code.visualstudio.com/Docs/editor/versioncontrol





