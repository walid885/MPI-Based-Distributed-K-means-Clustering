#!/bin/bash

# Enable color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display start message
echo -e "${BLUE}Starting the MPI-Based Distributed K-means Clustering process...${NC}"

# Run dataCleaning.py
echo -e "${YELLOW}Running data cleaning script...${NC}"
python3 ./src/dataCleaning.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Data cleaning completed successfully!${NC}"
else
    echo -e "${RED}Data cleaning script encountered an error.${NC}"
    exit 1
fi

# Run Comparison.py
echo -e "${YELLOW}Running comparison script...${NC}"
python3 ./src/Comparison.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Comparison completed successfully!${NC}"
else
    echo -e "${RED}Comparison script encountered an error.${NC}"
    exit 1
fi

# Display end message
echo -e "${BLUE}All processes completed successfully!${NC}"
