# catchMinor

Description of this library

## Example Code

## Installation

## Dependency

## Benchmark

## Implemented Algorithms

## Contribute
- gitflow & forkflow workflow
    - branch name: `master`, `develop`, `feature/{work}`, `release-{version}`, `hotfix`
```bash
# 1. fork repository
# 2. clone your origin and add remote upstream
git clone {your_origin_repo}
git add remote upstream {catchMinor_origin_repo}
# 3. make your origin develop branch as default branch
# 4. make feature branch
git checkout -b feature/{your_work} develop
# 5. do something to contribute
# 6. add, commit (use cz c)
git add {your_work}
git commit -m "{your_work_message}"
# 7. push to your origin (develop branch)
git push origin feature/{your_work}
# 8. PR in github
```
- merge
    - merge from feature branch to develop branch: `merge squash`
    - else: `merge --no-ff`