python39:
   pkgrepo.managed:
    - ppa: deadsnakes/ppa
   pkg.installed:
    - pkgs:
      - python3.10
      - python3.10-dev

python-packages:
   pip.installed:
      - names:
          - numpy >= 1.21.1
          - pandas >= 1.4.1
          - matplotlib >= 3.1.1
          - seaborn > 0.11.2
          - scipy >= 1.7.3
          - tqdm > =4.63.0
          - scikit-learn >= 1.0.2
          - flask >= 2.1.1
          - gdown >= 4.40
      - bin-env: "/usr/bin/python3.10 -m pip"

