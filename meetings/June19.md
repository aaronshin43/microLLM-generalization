# June 19 Meeting Notes

move on from binary classification of multiple targets / non-targets.
[Optional: experiment with large num. of targets e.g. 50 or 100?]

## possible next steps:

**non-binary classification** 
- output the target if present or 'n' otherwise.

**Count the number of targets present**

**position-dependent task**
- e.g. output the targets in the order they appear.
(or maybe binary e.g. 'y' if 's' occurs before 't', 'n' otherwise).

**longer outputs**
- e.g., if $r, s, t$ are targets and $u$ is non-target, then:
    - `"usuutun"` $\mapsto$ `":st"`
    - `"uuu"` $\mapsto$ `":n"`
    - `"uuuuturuuusun"` $\mapsto$ `":trs"`
