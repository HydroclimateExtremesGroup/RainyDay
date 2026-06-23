rm(list = ls()) 
library(ggplot2)


# 24  hour-basic case
# Atlas 14 data:
idf=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_24hr.idf',header=T,sep=",")  

# RainyDay data:
RD=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample/RainyDay_MadisonExample_FreqAnalysis_24hr.csv',header=T,sep=",")

txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=idf,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=idf,size=1,color="red")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_24hr.png',width = 8, height = 6.5, units = "in",dpi=300)




# 6  hour-basic case
# Atlas 14 data:
idf=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr.idf',header=T,sep=",")  

# RainyDay data:
RD=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")


p2<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=idf,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=idf,size=1,color="red")+  
  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p2

ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr.png',width = 8, height = 6.5, units = "in",dpi=300)



# 24  hour-basic case-20 storms
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample/RainyDay_MadisonExample_FreqAnalysis_24hr.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_20storms/RainyDay_MadisonExample_24hr_20storms_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
#ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_24hr.png',width = 8, height = 6.5, units = "in",dpi=300)


# 24  hour-basic case-nonuniform transpo
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample/RainyDay_MadisonExample_FreqAnalysis_24hr.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_Nonuniform/RainyDay_MadisonExample_24hr_Nonuniform_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1



# 24  hour-basic case-empirical resampling
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample/RainyDay_MadisonExample_FreqAnalysis_24hr.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_Empirical/RainyDay_MadisonExample_24hr_Empirical_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1


# 24  hour-basic case-no DurationCorrection
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample/RainyDay_MadisonExample_FreqAnalysis_24hr.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_NoCorrection/RainyDay_MadisonExample_24hr_NoCorrection_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1


# 6  hour-basic case-no DurationCorrection
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_NoCorrection/RainyDay_MadisonExample_6hr_NoCorrection_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1




# 24 hour-small domain vs large domain
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample/RainyDay_MadisonExample_FreqAnalysis_24hr.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain/RainyDay_MadisonExample_24hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_24hr_largeVsmall.png',width = 8, height = 6.5, units = "in",dpi=300)


# 6 hour-small domain vs large domain
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_LargeDomain/RainyDay_MadisonExample_6hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
txtsize=16
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr_largeVsmall.png',width = 8, height = 6.5, units = "in",dpi=300)




# 24 hour-large domain uniform vs. nonuniform tranposition
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain/RainyDay_MadisonExample_24hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain_nonuniform/RainyDay_MadisonExample_24hr_LargeDomain_nonuniform_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1

# 24 hour-large domain poisson vs. empirical
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain/RainyDay_MadisonExample_24hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain_empirical/RainyDay_MadisonExample_24hr_LargeDomain_empirical_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1


# 24 hour-large domain 40 vs 400 storms
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain/RainyDay_MadisonExample_24hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain_40storms/RainyDay_MadisonExample_24hr_LargeDomain_40storms_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1


# 24 hour-large domain AMS vs. PDS
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain/RainyDay_MadisonExample_24hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain_AMS/RainyDay_MadisonExample_24hr_LargeDomain_AMS_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1


# 24 hour-large domain no duration correction
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain/RainyDay_MadisonExample_24hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_24hr_LargeDomain_NoCorrection/RainyDay_MadisonExample_24hr_LargeDomain_NoCorrection_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1

# 6 hour-large domain no duration correction
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_LargeDomain/RainyDay_MadisonExample_6hr_LargeDomain_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_LargeDomain_NoCorrection/RainyDay_MadisonExample_6hr_LargeDomain_NoCorrection_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1





# 6 hour-original domain, 20 storms
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_20storms/RainyDay_MadisonExample_6hr_20storms_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr_20storms.png',width = 8, height = 6.5, units = "in",dpi=300)


# 6 hour-original domain, nonuniform
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_nonuniform/RainyDay_MadisonExample_6hr_nonuniform_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr_UniformVnonUniform.png',width = 8, height = 6.5, units = "in",dpi=300)


# 6 hour-original domain, empirical
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_empirical/RainyDay_MadisonExample_6hr_empirical_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr_empirical.png',width = 8, height = 6.5, units = "in",dpi=300)



# 6 hour-original domain, AMS
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr/RainyDay_MadisonExample_6hr_FreqAnalysis.csv',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_6hr_AMS/RainyDay_MadisonExample_6hr_AMS_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,400),breaks=c(0,100,200,300,400),labels=c("0","100","200","300","400"))+
  coord_cartesian(ylim=c(0,400),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_6hr_AMSvPDS.png',width = 8, height = 6.5, units = "in",dpi=300)


# 1 hour-original domain, vs atlas 14
RD_orig=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_1hr.idf',header=T,sep=",")
RD_new=read.table('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Examples/Madison/RainyDay_MadisonExample_1hr/RainyDay_MadisonExample_1hr_FreqAnalysis.csv',header=T,sep=",")
p1<-ggplot()+
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="dodgerblue",data=RD_orig,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_orig,size=1,color="dodgerblue")+  
  geom_ribbon(aes(x=returnperiod,ymin=minrain,ymax=maxrain),fill="red",data=RD_new,alpha=0.25, linetype=2)+
  geom_line(aes(x=returnperiod,y=meanrain),data=RD_new,size=1,color="red")+  
  scale_x_continuous(name="Exceedance Probability [-]\nReturn Period [y]",trans = 'log10',limits=c(2,500), breaks=c(2,5,10,20,50,100,200,500),labels=c("0.5\n2","0.2\n5","0.1\n10","0.05\n20","0.02\n50","0.01\n100","0.005\n200","0.002\n500"))+ 
  scale_y_continuous(name="Rainfall [mm]",limits=c(0,200),breaks=c(0,50,100,150,200),labels=c("0","50","100","150","200"))+
  coord_cartesian(ylim=c(0,200),xlim=c(2,500))+
  theme_bw(base_size = txtsize)+
  theme(panel.border=element_rect(size=1, colour = "black"),panel.grid.minor=element_blank(),panel.grid.major=element_line(size=0.25, colour = "grey"),axis.text=element_text(size=txtsize),axis.title=element_text(size=txtsize),plot.margin = unit(c(1,1,1,1), "cm"),legend.position="bottom")
p1
ggsave('/Users/daniel/Google_Drive/RainyDay2/RainyDayTutorials/Images/Madison_1hr.png',width = 8, height = 6.5, units = "in",dpi=300)

RD_orig$meanrain/RD_new$meanrain
