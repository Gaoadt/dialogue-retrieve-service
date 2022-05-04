python39:
   pkgrepo.managed:
    - ppa: deadsnakes/ppa
   pkg.installed:
    - pkgs:
      - python3.9
      - python3.9-dev

python-packages:
   pip.installed:
      - requirements: df/vagradfnt/requirement.txt
      - bin-env: "/usr/bin/python3.9 -m pip"
   require:
      - pkg: python3.9-pip

