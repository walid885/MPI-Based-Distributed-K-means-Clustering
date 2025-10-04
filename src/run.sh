#!/bin/bash

# Enhanced MPI-Based K-means Clustering with Comprehensive Benchmarking
# Author: [Your Name]
# Date: $(date +%Y-%m-%d)

# Enable color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NUM_PROCESSES=4  # Adjust based on your CPU count
PROCESS_CONFIGS="2 4"  # Test with 2 and 4 processes

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="benchmark_log_${TIMESTAMP}.txt"

# Function to log messages
log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        log_message "${GREEN}✓ $1 completed successfully!${NC}"
        return 0
    else
        log_message "${RED}✗ $1 encountered an error.${NC}"
        exit 1
    fi
}

# Display header
clear
log_message "${BLUE}════════════════════════════════════════════════════════════════${NC}"
log_message "${BLUE}    MPI-Based K-means Clustering - Comprehensive Benchmark${NC}"
log_message "${BLUE}════════════════════════════════════════════════════════════════${NC}"
log_message "${CYAN}Started: $(date)${NC}"
log_message "${CYAN}Log file: $LOG_FILE${NC}"
log_message ""

# Check prerequisites
log_message "${YELLOW}[1/7] Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_message "${RED}Python3 not found. Please install Python 3.7+${NC}"
    exit 1
fi
log_message "  ✓ Python3: $(python3 --version)"

# Check MPI
if ! command -v mpirun &> /dev/null; then
    log_message "${RED}mpirun not found. Please install OpenMPI or MPICH${NC}"
    exit 1
fi
log_message "  ✓ MPI: $(mpirun --version | head -1)"

# Check CPU count
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
log_message "  ✓ Available CPUs: $CPU_COUNT"

# Check dataset
if [ ! -f "../AstroDataset/star_classification.csv" ]; then
    log_message "${RED}Dataset not found at ../AstroDataset/star_classification.csv${NC}"
    exit 1
fi
DATASET_SIZE=$(du -h ../AstroDataset/star_classification.csv | cut -f1)
log_message "  ✓ Dataset found: $DATASET_SIZE"

log_message ""

# Phase 1: Data Cleaning & Preprocessing
log_message "${YELLOW}[2/7] Data Cleaning & Preprocessing...${NC}"
START_TIME=$(date +%s)

if [ -f "dataCleaning.py" ]; then
    python3 dataCleaning.py
    check_status "Data cleaning"
else
    log_message "${MAGENTA}  ⊙ dataCleaning.py not found, skipping...${NC}"
fi

CLEAN_TIME=$(($(date +%s) - START_TIME))
log_message "  Time: ${CLEAN_TIME}s"
log_message ""

# Phase 2: Dataset Analysis & Insights
log_message "${YELLOW}[3/7] Dataset Analysis & EDA...${NC}"
START_TIME=$(date +%s)

if [ -f "insights_analysis.py" ]; then
    python3 insights_analysis.py
    check_status "Dataset analysis"
else
    log_message "${MAGENTA}  ⊙ Using original script...${NC}"
    python3 insights.py 2>/dev/null || log_message "${MAGENTA}  ⊙ No insights script found${NC}"
fi

INSIGHTS_TIME=$(($(date +%s) - START_TIME))
log_message "  Time: ${INSIGHTS_TIME}s"
log_message ""

# Phase 3: Sequential K-means (Baseline)
log_message "${YELLOW}[4/7] Sequential K-means Baseline...${NC}"
START_TIME=$(date +%s)

python3 kmeans_sequantial.py
check_status "Sequential K-means"

SEQ_TIME=$(($(date +%s) - START_TIME))
log_message "  Execution time: ${SEQ_TIME}s"
log_message ""

# Phase 4: Parallel K-means (Multiple Configurations)
log_message "${YELLOW}[5/7] Parallel K-means Clustering...${NC}"

for PROCS in $PROCESS_CONFIGS; do
    if [ $PROCS -le $CPU_COUNT ]; then
        log_message "  ${CYAN}Testing with $PROCS processes...${NC}"
        START_TIME=$(date +%s)
        
        mpirun -n $PROCS python3 kmeans_parallel.py
        check_status "Parallel K-means ($PROCS processes)"
        
        PAR_TIME=$(($(date +%s) - START_TIME))
        SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $SEQ_TIME/$PAR_TIME}")
        log_message "  Execution time: ${PAR_TIME}s (Speedup: ${SPEEDUP}x)"
    else
        log_message "  ${MAGENTA}⊙ Skipping $PROCS processes (exceeds CPU count)${NC}"
    fi
done
log_message ""

# Phase 5: Performance Comparison
log_message "${YELLOW}[6/7] Generating Performance Comparison...${NC}"
START_TIME=$(date +%s)

python3 Comparison.py
check_status "Performance comparison"

COMP_TIME=$(($(date +%s) - START_TIME))
log_message "  Time: ${COMP_TIME}s"
log_message ""

# Phase 6: Generate Resume Metrics Report
log_message "${YELLOW}[7/7] Generating Resume Metrics Report...${NC}"
START_TIME=$(date +%s)

if [ -f "resume_metrics_generator.py" ]; then
    python3 resume_metrics_generator.py
    check_status "Resume metrics generation"
    
    METRICS_TIME=$(($(date +%s) - START_TIME))
    log_message "  Time: ${METRICS_TIME}s"
else
    log_message "${MAGENTA}  ⊙ resume_metrics_generator.py not found, skipping...${NC}"
fi
log_message ""

# Summary Report
TOTAL_TIME=$((CLEAN_TIME + INSIGHTS_TIME + SEQ_TIME + PAR_TIME + COMP_TIME + METRICS_TIME))

log_message "${GREEN}════════════════════════════════════════════════════════════════${NC}"
log_message "${GREEN}                    BENCHMARK COMPLETED!${NC}"
log_message "${GREEN}════════════════════════════════════════════════════════════════${NC}"
log_message ""
log_message "Execution Summary:"
log_message "  Total runtime: ${TOTAL_TIME}s ($(awk "BEGIN {printf \"%.1f\", $TOTAL_TIME/60}") minutes)"
log_message "  Sequential time: ${SEQ_TIME}s"
log_message "  Best parallel time: ${PAR_TIME}s"
if [ $SEQ_TIME -gt 0 ] && [ $PAR_TIME -gt 0 ]; then
    BEST_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $SEQ_TIME/$PAR_TIME}")
    log_message "  Best speedup: ${BEST_SPEEDUP}x"
fi
log_message ""

log_message "Generated Output:"
[ -d "insights_plots" ] && log_message "  ✓ insights_plots/ - $(ls insights_plots 2>/dev/null | wc -l) visualization files"
[ -d "kmeans_results" ] && log_message "  ✓ kmeans_results/ - $(ls kmeans_results 2>/dev/null | wc -l) clustering results"
[ -d "comparison_results" ] && log_message "  ✓ comparison_results/ - $(ls comparison_results 2>/dev/null | wc -l) comparison charts"
[ -d "benchmarks" ] && log_message "  ✓ benchmarks/ - $(ls benchmarks 2>/dev/null | wc -l) JSON metric files"
[ -d "resume_metrics" ] && log_message "  ✓ resume_metrics/ - Resume-ready reports"
log_message ""

# Display key metrics if available
if [ -f "resume_metrics/QUICK_REFERENCE.txt" ]; then
    log_message "${CYAN}═══ KEY METRICS FOR RESUME ═══${NC}"
    cat resume_metrics/QUICK_REFERENCE.txt | tee -a "$LOG_FILE"
    log_message ""
fi

log_message "${BLUE}Next Steps:${NC}"
log_message "  1. Review resume_metrics/QUICK_REFERENCE.txt for key numbers"
log_message "  2. Check resume_metrics/RESUME_METRICS_REPORT_*.txt for bullet points"
log_message "  3. View visualizations in insights_plots/ and comparison_results/"
log_message "  4. Use specific numbers from the reports in your CV"
log_message ""
log_message "${CYAN}Completed: $(date)${NC}"
log_message "${GREEN}════════════════════════════════════════════════════════════════${NC}"

exit 0