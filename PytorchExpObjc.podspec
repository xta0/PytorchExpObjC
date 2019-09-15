Pod::Spec.new do |s|
    s.name             = 'PytorchExpObjC'
    s.version          = '0.0.2'
    s.authors          = 'xta0'
    s.license          = { :type => 'MIT' }
    s.homepage         = 'https://github.com/xta0/PytorchExpObjC.git'
    s.source           = { :git => 'https://github.com/xta0/PytorchExpObjC.git', :branch => "master" }
    s.summary          = 'PytorchExp for Objective-C'
    s.description      = <<-DESC
   Objective-C wrapper of PytorchExp
                         DESC
  
    s.ios.deployment_target = '12.0'
    s.module_name = 'PytorchExpObjC'
    s.static_framework = true
    s.public_header_files = 'apis/*.h'
    s.source_files = [ 'apis/*.{h,m,mm}', 'src/*.{h,m,mm}' ]
    s.module_map = 'src/framework.modulemap'
    s.dependency 'LibTorch'
    s.pod_target_xcconfig = { 
      'HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_ROOT}/LibTorch/install/include" "${PODS_ROOT}/PytorchExpObjC/apis"',
      'VALID_ARCHS' => 'x86 arm64' 
    }
    s.library = 'c++', 'stdc++'

    # s.test_spec 'Tests' do |ts| 
    #   ts.source_files = 'Tests/*.{h,mm,m}'
    #   ts.resources = ['Tests/models/*.pt']
    # end
  end