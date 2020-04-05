var genNum;
var CsvToHtmlTable;
function format_link(link){
    if (link)
      return "<a href='" + link + "' target='_blank'>" + link + "</a>";
    else
      return "";
  }
function validateForm() {
    // Storing Field Values In Variables
    genNum = document.getElementById("genNum").value;
    console.log(genNum);
    var div = document.createElement('div');
    div.id = 'table-container'
    div.className = 'table-container'
    document.getElementsByTagName('body')[0].appendChild(div);
    
    var innerdiv = document.createElement('div');
    innerdiv.className = 'table-container'
    div.appendChild(innerdiv);
    console.log(div);
    console.log(innerdiv);
    //You have the table object
    
    //but you need to SHOW the object 

    $("#barn").append("<div id='table-container'><p>poo</p></div>");
    
    var table = new CsvToHtmlTable('data/data_reduc.csv', 'table-container', true,  {separator: ',', delimiter: '"'}, {"paging": false}, [[4, format_link]])
    var actualtable = table.giveTable(); // Html Table

    
    // CsvToHtmlTable.init({
    //     csv_path: 'data/data_reduc.csv',
    //     element: 'table-container', 
    //     allow_download: true,
    //     csv_options: {separator: ',', delimiter: '"'},
    //     datatables_options: {"paging": false},
    //     custom_formatting: [[4, format_link]]
    //   });
}

var CsvToHtmlTable = {
    init: function (options) {
        options = options || {};
        var csv_path = options.csv_path || "";
        var el = options.element || "table-container";
        var allow_download = options.allow_download || true;
        var csv_options = options.csv_options || {};
        var datatables_options = options.datatables_options || {};
        var custom_formatting = options.custom_formatting || [];
        var customTemplates = {};
        $.each(custom_formatting, function (i, v) {
            var colIdx = v[0];
            var func = v[1];
            customTemplates[colIdx] = func;
        });
        console.log(genNum)
        var $table = $("<table class='table table-borderless table-striped table-earning' id='" + el + "-table'></table>");
        var $containerElement = $("#" + el);
        $containerElement.empty().append($table);

        $.when($.get(csv_path)).then(
            function (data) {
                var csvData = $.csv.toArrays(data, csv_options);
                var $tableHead = $("<thead></thead>");
                var csvHeaderRow = csvData[0];
                var $tableHeadRow = $("<tr></tr>");
                for (var headerIdx = 0; headerIdx < csvHeaderRow.length; headerIdx++) {
                    $tableHeadRow.append($("<th></th>").text(csvHeaderRow[headerIdx]));
                }
                $tableHead.append($tableHeadRow);

                $table.append($tableHead);
                var $tableBody = $("<tbody></tbody>");
                
                for (var rowIdx = 1; rowIdx < genNum; rowIdx++) {
                    var $tableBodyRow = $("<tr></tr>");
                    for (var colIdx = 0; colIdx < csvData[rowIdx].length; colIdx++) {
                        var $tableBodyRowTd = $("<td></td>");
                        var cellTemplateFunc = customTemplates[colIdx];
                        if (cellTemplateFunc) {
                            $tableBodyRowTd.html(cellTemplateFunc(csvData[rowIdx][colIdx]));
                        } else {
                            $tableBodyRowTd.text(csvData[rowIdx][colIdx]);
                        }
                        $tableBodyRow.append($tableBodyRowTd);
                        $tableBody.append($tableBodyRow);
                    }
                }
                $table.append($tableBody);

                $table.DataTable(datatables_options);

                if (allow_download) {
                    $containerElement.append("<p><a class='btn btn-info' href='" + csv_path + "'><i class='glyphicon glyphicon-download'></i> Download as CSV</a></p>");
                }
            });
    }

    giveTable: function()
    {
        return $table;
    }
};
