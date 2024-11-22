#!/usr/bin/env bash
#Downloads data for the jupyter notebooks. Will take ~7GB of space

# Usage: ./get_data.sh [miniscope bpae mmf]
# If no arguments are given, all files are downloaded

mkdir rdm_data
cd rdm_data

file1_url="https://berkeley.box.com/shared/static/xdwvwud9mqrn7if03k213ubfwdlx99ir.gz"
file2_url="https://berkeley.box.com/shared/static/8dk1wxomlbwjc2iszb1odukaqlo0wmon.gz"
file3_url="https://berkeley.box.com/shared/static/ggwa3iiz8wi7z4wf4i2sl7lchqmgnnyf.gz"
file4_url="https://berkeley.box.com/shared/static/e2rvhlf5oylc3wemubxtev5k0oh7jl3k.gz"

if [ $# -eq 0 ]; then
    wget -v -O miniscope.tar.gz -L "$file1_url"
    tar -xzf miniscope.tar.gz
    rm miniscope.tar.gz

    wget -v -O multicolor.tar.gz -L "$file2_url"
    tar -xzf multicolor.tar.gz
    rm multicolor.tar.gz

    wget -v -O mmf.tar.gz -L "$file3_url"
    tar -xzf mmf.tar.gz
    rm mmf.tar.gz

    wget -v -O light-sheet.tar.gz -L "$file4_url"
    tar -xzf light-sheet.tar.gz
    rm light-sheet.tar.gz

else
    for arg in "$@"; do
        case $arg in
            "miniscope")
                wget -v -O miniscope.tar.gz -L "$file1_url"
                tar -xzf miniscope.tar.gz
                rm miniscope.tar.gz
                ;;
            "multicolor")
                wget -v -O multicolor.tar.gz -L "$file2_url"
                tar -xzf multicolor.tar.gz
                rm multicolor.tar.gz
                ;;
            "mmf")
                wget -v -O mmf.tar.gz -L "$file3_url"
                tar -xzf mmf.tar.gz
                rm mmf.tar.gz
                ;;
            "light-sheet")
                wget -v -O light-sheet.tar.gz -L "$file4_url"
                tar -xzf light-sheet.tar.gz
                rm light-sheet.tar.gz
                ;;
            *)
                echo "Invalid argument: $arg"
                ;;
        esac
    done
fi

