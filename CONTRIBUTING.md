# Contributor guide

## The basic rules

### Committing code
* Do not commit directly to `master`.
* Create new branches for each separate new feature or bugfix.
* Avoid making widespread changes to the code. If you can't avoid it, inform the other developers.
* Do **not** merge `master` into your local branch. Always [rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) instead. See [Avoiding automatic merges on pull](#avoiding-automatic-merges-on-pull) below.
* Once you're done with your changes, open a merge/pull request. Rebase again if necessary.

### Code style
* Make sure your editor or IDE supports EditorConfig. See "No Plugin Necessary" and "Download a Plugin" on https://editorconfig.org/.
* Before opening a merge/pull request, pass all your code through a linter. [Flake8](https://flake8.pycqa.org/en/latest/) is recommended.
* Avoid references to local paths and don't assume a specific operating system.
* Ideally, make everything 100% reproducible, using OS-independent functions, relative paths to files in the Git repo or publicly available URLs.

## More tips
* If the task allows it, create several smaller merge requests, each with a specific subproblem solved (or at least keep such subtasks in separate commits within one merge request). One merge request can set to depend on another, so the interdependencies can be tracked easily. Having smaller and simpler merge requests makes reviewing them easier and reduces the possibility for unnoticed mistakes.
* When creating branches for the merge requests, try using informative names, e.g. `fix/238` or `enh/238` (where `238` is an issue number) or `faster_csv_loading_v2` (where `v2` signifies that this is the second attempt to solve this problem, after the first merged `faster_csv_loading` branch hadn't achieved the expected result). As a rule, avoid using meaningless, generic names like `my_branch`, `fixes`, or `new_features`.
* After a merge request branch has been merged successfully, delete it. GitLab allows this to be automated: set the `Delete source branch when merge request is accepted.` option when creating the merge request. Don't let unnecessary branches accumulate.
* When working locally, **do not** commit directly to your local `master` branch or any other major branch if such exist (e.g. `develop`, `staging`). Do not locally commit also to branches from other people's merge requests, unless you've coordinated this with them and are going to push your changes in a synchronized matter (that is, nobody else will be pushing to the same branch at the same time). In general, only commit to the branches that you have created yourself.
* **IMPORTANT:** If you have committed changes locally and a merge request produces a conflict, do not merge the remote changes into your local branch if asked, and absolutely **do not push** such merge commits back to the remote repository! See below on how to turn off the automatic merges.

## Creating merge requests

GitLab already has an extensive manual on the topic: https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html

## Avoiding automatic merges on pull

To avoid the undesired automatic merge, set the following Git option in one of the two ways:

* for this repo only:
```sh
git config pull.ff only
```

* for all Git repos:
```sh
git config --global pull.ff only
```

After setting the option, you may see the following message when pushing local commits:
```
fatal: Not possible to fast-forward, aborting.
```

This means there are remote commits that you don't have locally. To attempt automatic rebasing:
```
git pull --rebase
```

If there are conflicting changes, you will need to first resolve the conflicts (see [git rebase](https://git-scm.com/docs/git-rebase)).

### Notes
If the above steps aren't carried out, recent versions of Git produce the following warning:
```
hint: Pulling without specifying how to reconcile divergent branches is
hint: discouraged. You can squelch this message by running one of the following
hint: commands sometime before your next pull:
hint:
hint:   git config pull.rebase false  # merge (the default strategy)
hint:   git config pull.rebase true   # rebase
hint:   git config pull.ff only       # fast-forward only
hint:
hint: You can replace "git config" with "git config --global" to set a default
hint: preference for all repositories. You can also pass --rebase, --no-rebase,
hint: or --ff-only on the command line to override the configured default per
hint: invocation.
```
