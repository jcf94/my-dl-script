#!/bin/sh

LOGFILE=$1.log
ROUGH_LOGFILE=$1_rough.log
FILTERED_LOGFILE=$1_filterd.log
ROUGH_DOTFILE=$1_rough.dot

if [ -f $LOGFILE ]; then
    echo "Get Log File" ${LOGFILE}
    awk 'match($0, /.*\|\|\s+(.*)/, out) {print out[1]}' $LOGFILE > $ROUGH_LOGFILE
    awk -f get_graph_filter.awk $ROUGH_LOGFILE > $FILTERED_LOGFILE
    awk -f get_graph.awk $FILTERED_LOGFILE > $ROUGH_DOTFILE
fi