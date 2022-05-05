# -*- mode: ruby -*-
# vi: set ft=ruby :

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://vagrantcloud.com/search.
  config.vm.box = "ubuntu/bionic64"

  # The url from where the 'config.vm.box' box will be fetched if it
  # doesn't already exist on the user's system.
  config.vm.box_url = "Vagrantfile"

  ## For masterless, mount your salt file root
  config.vm.synced_folder "configure/salt", "/srv/salt/"

  ## Use all the defaults:
  config.vm.provision :salt do |salt|
    salt.masterless = true
    salt.verbose = true
    salt.minion_config = "configure/salt/minion"
    salt.run_highstate = true

  end

  config.vm.network "forwarded_port", guest: 6000, host: 8080
end
