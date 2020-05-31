for Kaggle M5 competition Accuracy
Sharing scripts, preprocessing, models, training, submit

large_files:
https://drive.google.com/drive/folders/1NltzL-MqnKU7EJdAsmASDj2mznhIKgBX?usp=sharing
submit script:
https://www.kaggle.com/pedromoya/submit-motor

Local development/test/production environment:
Machine

    saul
        description: Computer
        width: 64 bits
        capabilities: smp
      *-core
           description: Motherboard
           physical id: 0
         *-memory
              description: System memory
              physical id: 0
              size: 23GiB
         *-cpu
              CPU AMD Ryzen 5 2600x 3.6 GHz 
              RAM 24 GB
              product: AMD Ryzen 5 2600X Six-Core Processor
              vendor: Advanced Micro Devices [AMD]
              physical id: 1
              bus info: cpu@0
              capacity: 3600MHz
              width: 64 bits
              capabilities: fpu fpu_exception wp vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp x86-64 pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave osxsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni cpufreq
      *-network
           description: Ethernet interface
           physical id: 1
           logical name: eth0
           serial: 00:d8:61:74:11:6e
           capabilities: ethernet physical
           configuration: broadcast=yes ip=xx.xx.xx.xx multicast=yes
           
    GPU
    
        szDescription	NVIDIA GeForce GTX 1660 SUPER 6 GB
        szDeviceId	0x21C4
        szDeviceIdentifier	{D7B71E3E-6284-11CF-6067-56E71BC2D735}
        szDeviceName	\\.\DISPLAY1
        szDisplayMemoryEnglish	18253 MB
        szDisplayMemoryLocalized	18253 MB
        szDisplayModeEnglish	1920 x 1080 (32 bit) (75Hz)
        szDisplayModeLocalized	1920 x 1080 (32 bit) (75Hz)
        szDriverAssemblyVersion	26.21.14.4274
        szDriverAttributes	Final Retail
        szDriverDateEnglish	12/03/2020 08:00:00 p. m.
        szDriverDateLocalized	3/12/2020 20:00:00
        szDriverLanguageEnglish	English
        szDriverLanguageLocalized	Inglés
        szDriverModelEnglish	WDDM 2.6
        szDriverModelLocalized	WDDM 2.6
        szDriverName	
        C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_f5de485bfda7bb25\nvldumdx.dll,
        C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_f5de485bfda7bb25\nvldumdx.dll,
        C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_f5de485bfda7bb25\nvldumdx.dll,
        C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_f5de485bfda7bb25\nvldumdx.dll
        szDriverNodeStrongName	oem3.inf:0f066de393ec985d:Section076:26.21.14.4274:pci\ven_10de&dev_21c4
        szDriverSignDate	Unknown
        szDriverVersion	26.21.0014.4274
        szKeyDeviceID	Enum\PCI\VEN_10DE&DEV_21C4&SUBSYS_C7581462&REV_A1
        szKeyDeviceKey	\Registry\Machine\System\CurrentControlSet\Control\Video\{164DEC55-6359-11EA-9498-806E6F6E6963}\0000
        szManufacturer	NVIDIA
    
    OS
    
        system='Windows', node='Saul', release='10', version='10.0.18362', machine='AMD64',
        processor='AMD64 Family 23 Model 8 Stepping 2, AuthenticAMD'
        Windows 10 Pro Version 1909 64 bits Activated ID product 00330-80129-24294-AA095
        IDE Pycharm Professional
            PyCharm 2019.3.4 (Professional Edition)
            Build #PY-193.6911.25, built on March 18, 2020
            Licensed to xxxx xxxxx
            Subscription is active until February 1, 2021
            Runtime version: 11.0.6+8-b520.43 amd64
            VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o
            Windows 10 10.0
            GC: ParNew, ConcurrentMarkSweep
            Memory: 1979M
            Cores: 12
            Registry: ide.windowSystem.autoShowProcessPopup=true, ide.balloon.shadow.size=0
            Non-Bundled Plugins: com.chrisrm.idea.MaterialThemeUI, ru.meanmail.plugin.requirements
        Python 3.8.2
        Windows App Ubuntu (subsystem)
            Editor Canonical Group limited
            Version 1804.2020.5.0
            Aplication 258 MB
            Data 1.07 GB
            Distributor ID: Ubuntu
                Description:    Ubuntu 18.04.4 LTS
                Release:        18.04
                Codename:       bionic

Remote development/test/production environment:




Version Control System
    
    local Git
    cloud repository Github


License 

    MIT License
    

How to train model



How to make predictions on a new test set



Side effects of the code



Key assumptions



Description of files in 1.0_configuration directory


