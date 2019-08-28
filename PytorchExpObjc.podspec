Pod::Spec.new do |s|
    s.name             = 'PytorchExpObjC'
    s.version          = '0.0.1'
    s.authors          = 'xta0'
    s.license          = { :type => 'MIT' }
    s.homepage         = 'https://github.com/xta0/PytorchExpObjC.git'
    s.source           = { :git => 'https://github.com/xta0/PytorchExpObjC.git', :branch => "master" }
    s.summary          = 'PytorchExp for Objective-C'
    s.description      = <<-DESC
   Objective-C wrapper of PytorchExp
                         DESC
  
    s.ios.deployment_target = '10.3'
    s.module_name = 'PytorchExpObjC'
    s.static_framework = true
    s.public_header_files = 'apis/*.h'
    s.source_files = [ 'apis/*.{h,m,mm}', 'src/*.{h,m,mm}' ]
    s.module_map = 'apis/framework.modulemap'
    s.dependency 'PytorchExp'
    s.pod_target_xcconfig = { 
      'HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_ROOT}/PytorchExp/install/include" "${PODS_ROOT}/PytorchExpObjC/apis"',
      'VALID_ARCHS' => 'armv7 armv7s arm64' 
    }
    s.library = 'c++', 'stdc++'
  end