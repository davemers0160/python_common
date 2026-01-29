# python_common/KICAD
These are the instructions to get python scripts integrated as action plugins into KICAD.  This has been tested on KICAD 9.0.7.

## File Structure
-- Plugin_Folder
    |
    -- __init__.py
	-- plugin_files.py
	-- plugin_button_icon(optional).png
	
## instructions
Copy the entire folder into the following typical location:

### Windows
%USER_DIRECTORY%/Documents/KiCad/9.0/scripting/plugins/

### Linux
~/.kicad/scripting/plugins
