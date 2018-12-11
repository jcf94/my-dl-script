#!/bin/awk

BEGIN {
    count = 0;
    print "digraph newgraph {\n";
}
{
    if (match($0, /(\w+)\s=\s(\w+)\[(.*)\]\((.*)\)/, out))
    {
        name = out[1];
        newname = sprintf("c%dn", count);
        gsub("n", newname, name);
        content = out[3];
        gsub("\"", "\\\"", content)
        printf("        %s[label=\"%s\", tooltip=\"%s\"];\n", name, out[2], content);
        if (out[4])
        {
            inpt = out[4];
            gsub("n", newname, inpt);
            printf("        %s -> %s;\n", inpt, name);
        }
    } else
    {
        print "#", $0;
        if (match($0, /\(.*\{/))
        {
            printf("    subgraph cluster_%d {\n        label=\"c%d\";\n", count, count);
            count ++;
        }
        else if (match($0, /\}/))
            print "    }";
    }
}
END {
    print "}";
}