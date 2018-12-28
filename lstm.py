import utils
import csv
import keras
import time
import datetime
import os
import multiprocessing

def main():
    my_data = utils.Nietzche_Data()
    mdl = utils.sl_lstm_model(my_data.chars, my_data.maxlen)
    num_epochs = 2
    data_size = 1000
    temperature = [0.2, 0.5, 1.0, 1.2]
    create_str_len = 10
    job_start_time = time.strftime("%Y%m%d_%H%M%S")
    data_directory = "data_" + job_start_time
    utils.nice_mk_dir(data_directory)
    Hardware_File = open(data_directory + "/hardware.txt","w")
    Hardware_File.write("CPU Count: " + round(multiprocessing.cpu_count() ,2))
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  
    mem_gib = mem_bytes/(1024.**3) 
    Hardware_File.write("Memory: " + round(mem_gib,2))
    Output_File = open(data_directory + "/output_" + job_start_time + ".txt","w")
    Output_File.write("=====================Begin: " + job_start_time + "======================\n")
    log_file = open(data_directory + "/logfile_" + job_start_time + ".csv","w")
    log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(['Job_Start_Time', 'Create_String_Len', 'Data_Size', 'Epoch_Num','Epoch_tm', 'Model_tm', \
        'SeedGen_tm', 'temp0.2_tm','temp0.5_tm','temp1.0_tm', 'temp1.2_tm'])
    for epoch in range(num_epochs):
      #EPOCH Start
      epoch_st = time.clock()
      if True:
        print("\n-------EPOCH: " + str(epoch) + "-------\n")
      Output_File.write("\n-------EPOCH: " + str(epoch) + "-------\n")
      #Fit Model for 1 epoch of available training data
      fit_model_st = time.clock()
      callbacks_list = [
          keras.callbacks.ModelCheckpoint(
              filepath=data_directory + '/my_model_{epoch}.h5'.format(epoch=epoch)
          )
      ]
      mdl.fit(my_data.x[0:data_size], my_data.y[0:data_size],
                batch_size=128,
                epochs=1,
                callbacks=callbacks_list,
                verbose=0
             )
      fit_model_et = time.clock()
      
      #Generate Seed Text
      seed_text_st = time.clock()
      seed_text = utils.get_seed_text(my_data.text, my_data.maxlen, print_output=False)
      Output_File.write("\nSeed Text: " + seed_text)
      seed_text_et = time.clock()
      
      #Generate Text
      generate_text_time = []
      for temp in temperature:
        generate_text_st = time.clock()
        generated_text = utils.generate_text(mdl, my_data.maxlen, my_data.chars, my_data.char_indices,
                                       seed_text, temp, create_str_len)
        generate_text_et = time.clock()
        generate_text_time.append(generate_text_et - generate_text_st)
        
        Output_File.write("\nGenerated Text: [Temp: {0}] {1}".format(temp, generated_text))
        Output_File.write("\n\n")
       
      epoch_et = time.clock()  
      epoch_time = round(epoch_et - epoch_st,3)
      model_time = round(fit_model_et - fit_model_st,3)
      seed_time = round(seed_text_et - seed_text_st,3)
      generate_text_time_formatted = ['%.3f' % elem for elem in generate_text_time]
      
      log_writer.writerow([job_start_time, create_str_len , data_size, epoch, epoch_time, model_time, seed_time,
                             generate_text_time_formatted[0], generate_text_time_formatted[1],
                             generate_text_time_formatted[2], generate_text_time_formatted[3]])
      
    Output_File.write("\nEnd Run: " + str(datetime.datetime.now()))  
    Output_File.write("\n\n")
        
    Output_File.close()
    
    log_file.close()

if __name__ == "__main__":
    main()
