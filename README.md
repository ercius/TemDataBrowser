# TemDataBrowser
A graphical user interface based on ScopeFoundry for viewing TEM data.

![GUI example](TemDataBrowser/images/TemDataBrowser_window.png)

# Installation
First install QT bindings. For example:

`$ pip install PyQt6`

Then install this package and the rest of the dependencies:

`$ pip install TemDataBrowser`

# Running the GUI

## From a python interpreter
Start python interpreter, import the package, and run the `open_file` function:

```
$ python
>>> import TemDataBrowser
>>> TemDataBrowser.main()
```

to start the graphical user interface.

## From the command line

A script should be installed on your system such that opening a command line
and typing `TemDataBrowser` will open the graphical uiser interface.
