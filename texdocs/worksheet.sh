#!/bin/bash

PRINTOUT=worksheet

while [ true ]
do
    case $1 in
	-algo*)
	    PRINTOUT=FlaAlgorithm;;
	-ws)
	    PRINTOUT=worksheet;;
	-wsnostep)
	    PRINTOUT=worksheetnosteps;;
	-wsupdate)
	    PRINTOUT=worksheetnosteps;;
	*)
	    break;;
    esac
    shift
done

NAME=$1


test -f $NAME || { echo "$NAME not found"; exit 1; }

TEMPNAME=$(tempfile -d . -s .tex)
TEXLOG=$(basename $TEMPNAME .tex).log
##echo $TEMPNAME $TEXLOG

cat >${TEMPNAME} <<EOF
\documentclass{article}
\input{packages}
\input{flame}
\input{flamisc}
\input{$NAME}
\begin{document}
\\$PRINTOUT
\end{document}
EOF

pdflatex --file-line-error-style --interaction batchmode ${TEMPNAME} >/dev/null 2>&1
stat=$?
if [ $stat -eq 0 ]
then
    mv -f $(basename ${TEMPNAME} .tex).pdf $(basename ${NAME} .tex).pdf
    rm -f $(basename ${TEMPNAME} .tex).*
else
    mv -f ${TEXLOG} $(basename ${NAME} .tex).log
    test -f $(basename ${TEMPNAME} .tex).pdf && \
	mv -f $(basename ${TEMPNAME} .tex).pdf $(basename ${NAME} .tex).pdf
    rm -f $(basename ${TEMPNAME} .tex).*
fi

    
