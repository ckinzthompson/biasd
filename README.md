# BIASD
This repository contains the Bayesian Inference for the Analysis of Sub-temporal-resolution Data (BIASD) algorithm written in Python.
Additionally, it includes a GUI, and a version of the integrand in the BIASD likelihood function that is written in C for optimal performance.

# To Use Git:

### To get the repository
```bash
git clone git@gitlab.com:ckinztho/biasd.git
```
will get the git repository from gitlab and clone it to the local machine

### After changing/creating a file
```bash
git add filename.txt
git commit -m "this is the commit message"
```

### Remove a file
```bash
git rm filename.txt
git commit -m "the commit message"
```

### Push local to gitlab
```bash
git push origin branch_name_here
```

### Make new branch
```bash
git checkout -b new_branch_name_here
```

### Pull updates from gitlab
```bash
git pull origin branch_name_here
```
### Merge branch into master
```bash
git checkout master
git pull origin master
git merge test
```
git push origin master