# ACL / ARR style files

Vendored, not fetched at build time. The authoring host 403s on github raw,
ctan.org and tectonic's bundle server, so a build that reaches the network for
these is a build that only works on some machines.

Pinned to acl-org/acl-style-files @ `d5adc823ff0f80f98c80405ca0ab66c68e684409`
(see UPSTREAM_COMMIT.txt), obtained via bolt task `4xj597v7gu`. The pin matters
for more than provenance: an upstream change to `acl.sty` between now and the
submission build would silently change the page count, which is the one number
the ARR 8-page limit is enforced on.

  acl.sty         two-column ACL/ARR document style
  acl_natbib.bst  the bibliography style ARR expects
