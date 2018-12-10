#!/bin/awk

BEGIN {
    RS = "";
    FS = "\n";
    count = 0;
    list[0] = "";
}
{
    list[count] = $0;
    count ++;
}
END {
    asort(list)

    print list[0];
    for (i=1;i<count;i++)
    if (list[i] != list[i-1])
        print list[i];
}