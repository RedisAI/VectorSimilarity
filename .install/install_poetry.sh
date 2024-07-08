OS_TYPE=$(uname -s)
MODE=$1 # whether to install using sudo or not

curl -sSL https://install.python-poetry.org | python3 -


if [[ $OS_TYPE == Darwin ]]; then
    echo "export PATH='$HOME/.local/bin:$PATH'" >> ~/.zshrc
else
    echo "export PATH='$HOME/.local/bin:$PATH'" >> ~/.bash_profile
fi

poetry config virtualenvs.in-project true
