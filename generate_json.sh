#!/bin/bash
asetup Athena,main,latest
# ----------- Logging Colors -----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

# ----------- Argument Parsing -----------
if [ "$#" -ne 1 ]; then
    echo -e "${RED}❌ Usage: $0 <defaultLevel>${NC}"
    exit 1
fi

DEFAULT_LEVEL=$1
LEVEL_TAG="dl${DEFAULT_LEVEL}"  # e.g., dl10

echo -e "${CYAN}Using defaultLevel = ${DEFAULT_LEVEL}${NC}"
echo

# ----------- Ensure required directories exist -----------
mkdir -p /eos/user/y/yolanney/json_files/real
mkdir -p /eos/user/y/yolanney/json_files/sim
mkdir -p /eos/user/y/yolanney/compressed_files/real
mkdir -p /eos/user/y/yolanney/compressed_files/sim
mkdir -p logs

# ----------- Input Files -----------
REAL_FILES=(
"/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6479/data18_13TeV.00348885.physics_Main.deriv.DAOD_PHYSLITE.r13286_p4910_p6479/DAOD_PHYSLITE.41578717._000256.pool.root.1"
"/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6482/data23_13p6TeV.00456749.physics_Main.deriv.DAOD_PHYSLITE.r15774_p6304_p6482/DAOD_PHYSLITE.41588921._000002.pool.root.1"
)

SIM_FILES=(
"/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6490/mc20_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_PHYSLITE.e6337_s3681_r13167_r13146_p6490/DAOD_PHYSLITE.41651753._000007.pool.root.1"
"/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/ASG/DAOD_PHYSLITE/p6491/mc23_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.deriv.DAOD_PHYSLITE.e8514_e8528_s4162_s4114_r15540_r15516_p6491/DAOD_PHYSLITE.41633384._000941.pool.root.1"
)

# ----------- Output base directories -----------
JSON_BASE="/eos/user/y/yolanney/json_files"
COMPRESSED_BASE="/eos/user/y/yolanney/compressed_files"

# ----------- Process each file -----------
process_file() {
    input_file="$1"
    category="$2"

    filename=$(basename "$input_file")
    filename_noext="${filename%.*}"

    json_path="${JSON_BASE}/${category}/${filename_noext}_${LEVEL_TAG}.json"
    compressed_path="${COMPRESSED_BASE}/${category}/${filename_noext}_${LEVEL_TAG}_compressed.root"
    log_path="logs/${filename_noext}_${LEVEL_TAG}_compression.log"
    json_log_path="logs/${filename_noext}_${LEVEL_TAG}_json.log"

    echo -e "${CYAN}📄 Processing: $filename ($category)${NC}"

    # Step 1: JSON generation
    if [[ -f "$json_path" ]]; then
        echo -e "${GREEN}✅ JSON exists:${NC} $json_path"
    else
        echo -e "${YELLOW}🔧 Generating JSON...${NC}"
        python physlite-compression/GenerateJsonConfigFile.py \
            --inputFile "$input_file" \
            --outputJsonFile "$json_path" \
            --defaultLevel "$DEFAULT_LEVEL" \
            > "$json_log_path" 2>&1

        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✔️ JSON generated:${NC} $json_path"
            echo -e "${MAGENTA}📝 JSON log written to:${NC} $json_log_path"
        else
            echo -e "${RED}❌ JSON generation failed for:${NC} $filename"
            echo -e "${MAGENTA}📝 Check log:${NC} $json_log_path"
            return
        fi
    fi

    # Step 2: Compression
    if [[ -f "$compressed_path" ]]; then
        echo -e "${GREEN}✅ Compressed file already exists:${NC} $compressed_path"
    else
        echo -e "${BLUE}📦 Starting compression...${NC}"
        python physlite-compression/CompressFromJson.py \
            --inputFiles "$input_file" \
            --outputFile "$compressed_path" \
            --compressionConfig "$json_path" \
            --highMantissaBits 7 \
            --lowMantissaBits 15 \
            > "$log_path" 2>&1

        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✔️ Compression successful:${NC} $compressed_path"
            echo -e "${MAGENTA}📝 Log written to:${NC} $log_path"
        else
            echo -e "${RED}❌ Compression failed for:${NC} $filename"
            echo -e "${MAGENTA}📝 Check log:${NC} $log_path"
        fi
    fi

    echo -e "${CYAN}-----------------------------------------------------${NC}\n"
}

# ----------- Process real and sim files -----------
for file in "${REAL_FILES[@]}"; do
    process_file "$file" "real"
done

for file in "${SIM_FILES[@]}"; do
    process_file "$file" "sim"
done