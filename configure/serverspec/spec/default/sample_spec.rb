require 'spec_helper'


# Helper function developed for spec simplicity
def pip3_require(name, version) 
  describe package(name) do
    let(:disable_sudo) { true }
    it { should be_installed.by('pip3').with_version(version) }
  end
end


# Check python installation
describe package('python3.9') do
  it { should be_installed }
end


pip3_require 'numpy', '1.22.3'
pip3_require 'pandas', '1.4.2'
pip3_require 'matplotlib', '3.5.2'
pip3_require 'seaborn', '0.11.2'
pip3_require 'scipy', '1.8.0'
pip3_require 'tqdm', '4.64.0'
pip3_require 'scikit-learn', '1.0.2'
pip3_require 'Flask', '2.1.2'
pip3_require 'gdown', '4.4.0'

