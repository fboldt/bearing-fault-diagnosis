#!/bin/bash

# Minimal group of libraries for framework functionality
install_minimal() {
    echo "Installing minimal dependencies..."
    pip install numpy scipy requests pyunpack rarfile scikit-learn imblearn PyWavelets
    echo "Minimal dependencies successfully installed."
}

# Group of libraries to run a CNN
install_cnn() {
    echo "Installing dependencies for CNN..."
    pip install tensorflow
    echo "Dependencies for CNN successfully installed."
}

main() {
    echo "Select installation options:"
    echo "1) Install minimal dependencies"
    echo "2) Install dependencies for CNN1D"
    echo "3) Install all dependencies"

    read -p "Option (1-3): " option

    case $option in
        1)
            install_minimal
            ;;
        2)
            install_minimal
            install_cnn
            ;;
        3)
            install_minimal
            install_cnn
            ;;
        *)
            echo "Invalid option. Please select an option between 1 and 3."
            ;;
    esac
}

main