FROM mcr.microsoft.com/mirror/docker/library/ubuntu:24.04

ENV PROJECT_NAME=scaling_laws_perturbations
ENV PROJECT_PATH=/workspaces/$PROJECT_NAME

WORKDIR $PROJECT_PATH

RUN <<EOF
# install system packages
set -eux
apt-get update
apt-get install -y --no-install-recommends \
  curl wget lsb-release git zsh build-essential sudo \
  file ca-certificates tmux fzf direnv neovim nano \
  build-essential

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/*
EOF

RUN <<EOF
# create linuxbrew user
useradd -m linuxbrew
mkdir -p /home/linuxbrew

chown -R ubuntu:ubuntu $PROJECT_PATH
chsh -s /usr/bin/zsh ubuntu
EOF

RUN <<EOF
# let ubuntu user run sudo commands
echo ubuntu ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/ubuntu
chmod 0440 /etc/sudoers.d/ubuntu
EOF

RUN <<EOF
# install linuxbrew
set -eux

su - linuxbrew -c '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
chown -R ubuntu:ubuntu /home/linuxbrew/
EOF

ENV PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:${PATH}"

# this user is created by the base 24.04 ubuntu image
USER ubuntu

RUN <<EOF
# install user-level brew packages
set -eux

# kubelogin in brew is the equivalent of krew oidc-login plugin
brew install gcc azure-cli kubectl kubelogin uv gh

# initialize az cli things, to hide odd warnings from the user later
az

rm -rf /home/linuxbrew/.cache/*
EOF

COPY --chown=ubuntu:ubuntu .devcontainer/.direnvrc      /home/ubuntu/.direnvrc
COPY --chown=ubuntu:ubuntu .devcontainer/.config/direnv /home/ubuntu/.config/direnv

ENV PATH="/home/ubuntu/.local/bin:${PATH}"

RUN <<EOF
# install oh-my-zsh for user
set -eux

sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
EOF

RUN <<EOF
# add direnv to bash/zsh
set -eux

echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
EOF

RUN <<EOF
# add uv+uvx completions to bash/zsh
set -eux

echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc

echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc
EOF

RUN <<EOF
# install pythons and tools using uv
set -eux

uv python install 3.10 3.12
# make a symlink for poetry to find our default python version
ln -s ~/.local/bin/python3.12 ~/.local/bin/python

uv tool install poetry
poetry config virtualenvs.in-project true

rm -rf ~/.cache/uv
EOF

RUN <<EOF
# configure common git options
set -eux

git config --global core.editor "nano"
git config --global pager.branch false
git config --global alias.ba branch
git config --global alias.co checkout
git config --global alias.st status
git config --global alias.sa stash
git config --global alias.cp cherry-pick
git config --global alias.mt mergetool
git config --global alias.pl pull --rebase
EOF

CMD ["zsh"]
