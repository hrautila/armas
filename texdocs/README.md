
Algorithm spesification in {name}.tex file. It defines following commands
that are used in worksheet/algorithm print outs. 

\operation  
   Routine name as used in worksheets. Command embedded in math-mode.
\routinename
   Routine name as used in algorithm printout. Command embedded in math-mode.
\precondition
   Precondition, embedded in math mode.
\postcondition
   Postcondition, embedded in math mode.
\guard
   Loop guard, embedded in math mode
\invariant
   Invariant, embedded in math mode
\partitionings
   Initial partitiong, not in math mode
\partitionsizes
   Initial partition size info, not in math mode
\blocksizeftex
   Blocksize in loop, in math mode
\repartitionings
   Top of loop repartitioning, not in math mode
\repartitionsizes
   Repartition sizes, not in math mode
\moveboundaries
   Bottom of loop repartitioning, not in math mode
\beforeupdate
   Before update state spec, not in math mode
\update
   Update operations, not in math mode
\afterupdate
   After update state, not in math mode

Algorithm spec file only defines above elements, nothing else. Printing out
single worksheet/algo is with worksheet.sh command that create temporary
LaTex document with proper context.
